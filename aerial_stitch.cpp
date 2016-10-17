
#include "util.h"
#include "match.h"
#include "warp.h"
#include "mst.h"
#include "exposure_compensator.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <time.h>


using namespace std;
using namespace cv;
using namespace cv::detail;



Rect compose_roi(vector<Mat> imgs, vector<Point> corners)
{
	int n = imgs.size();
	int x1, x2, y1, y2;
	x1 = corners[0].x;
	x2 = corners[0].x + imgs[0].cols;
	y1 = corners[0].y;
	y2 = corners[0].y + imgs[0].rows;
	for(int i = 1; i < n; i++)
	{
		x1 = (std::min)(corners[i].x, x1);
		y1 = (std::min)(corners[i].y, y1);
		x2 = (std::max)(corners[i].x + imgs[i].cols, x2);
		y2 = (std::max)(corners[i].y + imgs[i].rows, y2);
	}
	return Rect(x1, y1, x2-x1, y2-y1);
}

Point compose_images_rough(vector<Mat> imgs, vector<Mat> masks, vector<Point> corners, Mat &dst, Mat &dst_mask)
{
	int n = imgs.size();
	Rect roi = compose_roi(imgs, corners);
	Point corner = roi.tl();
	dst.create(roi.size(), CV_8UC3);
	dst_mask.create(roi.size(), CV_8U);
	for(int i = 0; i < n; i++)
	{
		int x = corners[i].x - corner.x;
		int y = corners[i].y - corner.y;
		imgs[i].copyTo(dst(Rect(x, y, imgs[i].cols, imgs[i].rows)), masks[i]);
		masks[i].copyTo(dst_mask(Rect(x, y, imgs[i].cols, imgs[i].rows)), masks[i]);
	}
	return corner;
}

Rect get_mask_roi(Mat mask)
{
	int x1, x2, y1, y2;
	x2 = y2 = 0;
	x1 = mask.cols - 1;
	y1 = mask.rows - 1;
	for(int y = 0; y < mask.rows; y++)
	{
		uchar *mask_ptr = mask.ptr<uchar>(y);
		for(int x = 0; x < mask.cols; x++)
		{
			if(mask_ptr[x] != 0)
			{
				if(x < x1) x1 = x;
				if(y < y1) y1 = y;
				if(x > x2) x2 = x;
				if(y > y2) y2 = y;
			}
		}
	}
	return Rect(x1, y1, x2-x1+1, y2-y1+1);
}

/*
	所有的特征提取、匹配都是在work_scale
*/
double work_scale = 0.4;
static int stitch_idx = 1;
string src_dir;
//"D:/data/my/huludao/1/undistort/";//"D:/data/my/plane/8-06_jingyuan/ec/";//"data/likelou/";
//"D:/data/my/plane/7-17_likelou/";//"D:/data/my/plane/8-08_zhongguanxinyuan/1/";

/*
	将srcs[1]~srcs[n]增量式拼接到srcs[0]
	srcs:	srcs[0]是基准图像，srcs[i]是增量图像
	mask0:	srcs[0]的mask
	feat0:	srcs[0]的特征
*/
int stitch_images(vector<Mat> srcs, Mat mask0, ImageFeatures feat0, 
	Mat &dst, Mat &dst_mask, ImageFeatures &dst_feat, vector<bool> &success_mask)
{
	int num_img = srcs.size();

	/***********          work scale             ***********/
	cout << "\textracting features ... " << endl;
	vector<Mat> srcs_ws(num_img);
	Mat mask0_ws;
	resize(mask0, mask0_ws, Size(), work_scale, work_scale);
	vector<ImageFeatures> feats(num_img);
	Ptr<FeaturesFinder> finder = new SurfFeaturesFinder();
	for(int i = 0; i < num_img; i++)
		resize(srcs[i], srcs_ws[i], Size(), work_scale, work_scale);

	//	是否开启增量式特征提取
	
	bool is_acc_feat = false;
	if(is_acc_feat)
	{
		//	如果feat0里面没有特征，就重新提取，否则直接使用
		if(feat0.keypoints.size() == 0)
			(*finder)(srcs_ws[0], feats[0]);
		else
			feats[0] = feat0;
		for(int i = 1; i < num_img; i++)
			(*finder)(srcs_ws[i], feats[i]);
	}
	else
	{
#pragma omp parallel for
		for(int i = 0; i < num_img; i++)
			(*finder)(srcs_ws[i], feats[i]);
		dst_feat = feats[0];
	}
	finder->collectGarbage();
	//	feature matching
	cout << "\tmatching the features ... " << endl;
	vector<MatchesInfo> matches_infos(num_img-1);
	BestOf2NearestMatcher matcher(false, 0.3f);
	//for(int i = 1; i < num_img; i++)
	//	feature_match_bidirection_raw(feats[0], feats[i], matches_infos[i-1]);
#pragma omp parallel for
	for(int i = 1; i < num_img; i++)
		//feature_match_bidirection_raw(feats[0], feats[i], matches_infos[i-1]);
		//matcher(feats[0], feats[i], matches_infos[i-1]);
		feature_match_bidirection(feats[0], feats[i], matches_infos[i-1]);
	matcher.collectGarbage();
	char img_name[100];
	Mat match_img;
	for(int i = 1; i < num_img; i++)
	{
		my_drawMatches2(srcs_ws[0], srcs_ws[i], feats[0], feats[i], matches_infos[i-1], match_img);
		sprintf(img_name, "debug/match_%d_%d.jpg", stitch_idx, i);
		imwrite(src_dir + img_name, match_img);
	}

	//	motion estimation: compute scale, theta and tx, ty
	cout << "\tmotion estimation ... " << endl;
	vector<Point> corners;
	vector<Mat> imgs_warp_ws, masks_warp_ws, Hs;
	Hs.push_back(Mat());
	imgs_warp_ws.push_back(srcs_ws[0]);
	masks_warp_ws.push_back(mask0_ws);
	corners.push_back(Point(0, 0));
	success_mask.clear();
	success_mask.push_back(true);
	for(int i = 1; i < num_img; i++)
	{
		Mat H_tmp, img_warp_ws_tmp, mask_warp_ws_tmp;
		//if(compute_perspective_transform(feats[0], feats[i], matches_infos[i-1], H_tmp) == -1)
		if(compute_transform(feats[0], feats[i], matches_infos[i-1], H_tmp, AS_SIMILAR) == -1)
		{
			cout << "\tERROR: image " << i << " stitch error" << endl;
			success_mask.push_back(false);
			continue;
		}
		Rect dst_rect = warp_area(H_tmp, Rect(0, 0, srcs_ws[i].cols, srcs_ws[i].rows));
		if(dst_rect.area() > 5760000)
		{
			cout << "\tERROR: image " << i << " stitch error" << endl;
			success_mask.push_back(false);
			continue;
		}
		Point corner_tmp = my_warp_perspective(srcs_ws[i], img_warp_ws_tmp, mask_warp_ws_tmp, H_tmp);
		corners.push_back(corner_tmp);
		imgs_warp_ws.push_back(img_warp_ws_tmp);
		masks_warp_ws.push_back(mask_warp_ws_tmp);
		Hs.push_back(H_tmp);
		success_mask.push_back(true);
	}
	//	剔除掉拼接失败的图像
	vector<Mat> srcs_succ;
	vector<ImageFeatures> feats_succ;
	for (int i = 0; i < num_img; ++i)
	{
		if(success_mask[i])
		{
			srcs_succ.push_back(srcs[i]);
			feats_succ.push_back(feats[i]);
		}
		else
		{
			srcs[i].release();
			feats[i].descriptors.release();
		}
	}
	srcs.clear();
	feats.clear();
	srcs = srcs_succ;
	feats = feats_succ;
	num_img = srcs.size();
	if(num_img < 2)
	{
		dst_feat = feat0;
		dst_mask = mask0;
		return 0;
	}

	//	downsample for compensate exposure
	cout << "\tcompensate exposure" << endl;
	vector<Mat> images_ec(num_img), masks_ec(num_img);
	vector<Mat_<float>> ec_weight_maps;
	vector<Point> ec_corners(num_img);
	double ec_scale = 0.5;
	for (int i = 0; i < num_img; ++i)
	{
		resize(imgs_warp_ws[i], images_ec[i], Size(), ec_scale, ec_scale);
		resize(masks_warp_ws[i], masks_ec[i], Size(), ec_scale, ec_scale);
		ec_corners[i] = corners[i] * ec_scale;
	}
	MyExposureCompensator compensator;
	compensator.createWeightMaps(ec_corners, images_ec, masks_ec, ec_weight_maps);

	/***********          seam scale             ***********/
	//	seam
	cout << "\tfind seam" << endl;
	vector<Mat> images_warped_forseam(num_img), masks_seam_ss(num_img);
	vector<Point> seam_corners(num_img);
	double seam_scale = 0.2;	//	downsample to accelerate
	for (int i = 0; i < num_img; ++i)
	{
		imgs_warp_ws[i].convertTo(images_warped_forseam[i], CV_32F);
		resize(images_warped_forseam[i], images_warped_forseam[i], Size(), seam_scale, seam_scale);
		resize(masks_warp_ws[i], masks_seam_ss[i], Size(), seam_scale, seam_scale);
		imshow("test", images_warped_forseam[i]);
		imshow("test1", masks_seam_ss[i]);
		waitKey(0);
		seam_corners[i] = corners[i] * seam_scale;
	}
	/*//	两图重叠部分，如果像素值大于阈值，则直接置为256*(i+1)
	for (int i = 1; i < num_img; ++i)
	{
		Point corner = seam_corners[0] - seam_corners[i];
		int left = max(0, corner.x);
		int right = min(images_warped_forseam[i].cols, corner.x + images_warped_forseam[0].cols);
		int up = max(0, corner.y);
		int down = min(images_warped_forseam[i].rows, corner.y + images_warped_forseam[0].rows);
		for (int y = up; y < down; y++)
		{
			Point3_<float> *seam_cur_row = images_warped_forseam[i].ptr<Point3_<float>>(y);
			Point3_<float> *seam_0_row = images_warped_forseam[0].ptr<Point3_<float>>(y-corner.y);
			for(int x = left; x < right; x++)
			{
				Point3_<float> delta_pix = seam_0_row[x-corner.x] - seam_cur_row[x];
				float error_pix = delta_pix.x * delta_pix.x + delta_pix.y * delta_pix.y + delta_pix.z * delta_pix.z;
				if (error_pix > 1200)
					;//seam_cur_row[x] = Point3_<float>(256*(i+1), 256*(i+1), 256*(i+1));
			}
		}
	}*/
	Ptr<SeamFinder> seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	seam_finder->find(images_warped_forseam, seam_corners, masks_seam_ss);

	/***********          work scale             ***********/
	//	增量式特征提取
	if (is_acc_feat)
	{
		vector<Mat> masks_seam_ws(num_img);
		vector<ImageFeatures> feats_dst_add(num_img);
		finder = new SurfFeaturesFinder();
		Rect dst_roi_ws = compose_roi(imgs_warp_ws, corners);
		for(int i = 0; i < num_img; i++)
		{
			Mat dilated_mask, seam_mask;
			dilate(masks_seam_ss[i], dilated_mask, Mat());
			resize(dilated_mask, seam_mask, masks_warp_ws[i].size());
			masks_seam_ws[i] = seam_mask & masks_warp_ws[i];
			masks_seam_ws[i].convertTo(masks_seam_ws[i], CV_8U);
		}
		//	依次提取warp后的特征
		for(int i = 1; i < num_img; i++)
		{
			Rect roi = get_mask_roi(masks_seam_ws[i]);
			Mat img_tmp = Mat::zeros(imgs_warp_ws[i].size(), CV_8UC3);
			imgs_warp_ws[i].copyTo(img_tmp);//, masks_seam_ws[i]);
			compensator.apply(i, img_tmp);
			(*finder)(img_tmp(roi), feats_dst_add[i]);
			//cout << feats_dst_add[i].keypoints.size() << "; " <<feats_dst_add[i].descriptors.cols << ", " << feats_dst_add[i].descriptors.rows << endl;
			for(int k = 0; k < feats_dst_add[i].keypoints.size(); k++)
				feats_dst_add[i].keypoints[k].pt += Point2f(roi.x, roi.y);

			sprintf(img_name, "debug/%d_%d.jpg", stitch_idx, i);
			rectangle(img_tmp, roi, Scalar(0, 0, 255, 0));
			imwrite(src_dir + img_name, img_tmp);
		}
		finder->collectGarbage();
		feats_dst_add[0] = feats[0];
		//	构造dst_feat
		dst_feat.img_size = dst_roi_ws.size();
		dst_feat.keypoints.clear();
		int dst_feat_num = 0;
		vector<Mat> dst_feat_descriptors;
		for(int i = 0; i < num_img; i++)
		{
			int feat_num = feats_dst_add[i].keypoints.size();
			uchar *mask_ptr = masks_seam_ws[i].ptr<uchar>(0);
			int mask_width = masks_seam_ws[i].cols;
			for(int j = 0; j < feat_num; j++)
			{
				Point p = feats_dst_add[i].keypoints[j].pt;
				if(mask_ptr[p.y * mask_width + p.x] != 0)
				{
					dst_feat.keypoints.push_back(feats_dst_add[i].keypoints[j]);
					dst_feat.keypoints[dst_feat_num++].pt += Point2f(corners[i].x - dst_roi_ws.x, corners[i].y - dst_roi_ws.y);
					dst_feat_descriptors.push_back(feats_dst_add[i].descriptors.row(j));
				}
			}
		}
		int descriptor_dim = dst_feat_descriptors[0].cols;
		dst_feat.descriptors.create(dst_feat_num, descriptor_dim, dst_feat_descriptors[0].type());
		for(int i = 0; i < dst_feat_num; i++)
			dst_feat_descriptors[i].copyTo(dst_feat.descriptors(Rect(0, i, descriptor_dim, 1)));
	}


	/***********        compose scale            ***********/
	//	Rescale the seam mask and Hs
	cout << "\tRescale" << endl;
	double compose_scale = 1, compose_work_aspect = compose_scale / work_scale;
	Mat mask0_cs;
	resize(mask0, mask0_cs, Size(), compose_scale, compose_scale);
	vector<Mat> imgs_cs(num_img), imgs_warped_cs(num_img);
	vector<Size> sizes_cs(num_img);
	vector<Mat> masks_cs(num_img), masks_warped_cs(num_img);
#pragma omp parallel for
	for(int i = 0; i < num_img; i++)
	{
		resize(srcs[i], imgs_cs[i], Size(), compose_scale, compose_scale);
		if(i == 0)
		{
			imgs_warped_cs[i] = imgs_cs[i];
			masks_warped_cs[i] = mask0_cs;
		}
		else
		{
			//	H and warp
			Hs[i].at<double>(0, 2) *= compose_work_aspect;
			Hs[i].at<double>(1, 2) *= compose_work_aspect;
			Hs[i].at<double>(2, 0) /= compose_work_aspect;
			Hs[i].at<double>(2, 1) /= compose_work_aspect;
			corners[i] = my_warp_perspective(imgs_cs[i], imgs_warped_cs[i], masks_warped_cs[i], Hs[i]);
		}
		sizes_cs[i] = imgs_warped_cs[i].size();
		compensator.apply(i, imgs_warped_cs[i]);

		//	seam and mask
		Mat dilated_mask, seam_mask;
		dilate(masks_seam_ss[i], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, masks_warped_cs[i].size());
		masks_cs[i] = seam_mask & masks_warped_cs[i];
	}
	//compensator.gainMapResize(sizes_cs, ec_weight_maps);

	//	blend
	/*
	cout << "\tblending ..." << endl;
	Rect dst_roi = compose_roi(imgs_warped_cs, corners);
	int blend_type = Blender::FEATHER;
	float blend_strength = 3;
	Ptr<Blender> blender = Blender::createDefault(blend_type, false);
	float blend_width = sqrt(static_cast<float>(dst_roi.area())) * blend_strength / 100.f;
	if (blend_type == Blender::MULTI_BAND)
	{
		MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
		mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
		cout << "\t\tMulti-band blender, number of bands: " << mb->numBands() << endl;
	}
	else if (blend_type == Blender::FEATHER)
	{
		FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
		fb->setSharpness(1.f/blend_width);
		cout << "\t\tFeather blender, sharpness: " << fb->sharpness() << endl;
	}
	blender->prepare(corners, sizes_cs);
	cout << "\t\tblending prepared ..." << endl;

	for(int i = 0; i < num_img; i++)
	{
		Mat img_warped_s;
		imgs_warped_cs[i].convertTo(img_warped_s, CV_16S);
		blender->feed(img_warped_s, masks_cs[i], corners[i]);
	}
	blender->blend(dst, dst_mask);
	dst.convertTo(dst, CV_8U);
	*/
	cout << "\tcompose" << endl;
	//imwrite("D:/data/my/plane/7-17_likelou/1/debug/warp.jpg", imgs_warped_cs[1]);
	compose_images_rough(imgs_warped_cs, masks_cs, corners, dst, dst_mask);
	mask0.release();
	srcs[0].release();
	stitch_idx++;
	return 0;
}

vector<string> image_set1(string &src_dir)
{
	src_dir = "D:/data/my/plane/8-06_jingyuan/1/";
	vector<string> img_names;
	img_names.push_back(src_dir + "DJI01270.jpg");
	img_names.push_back(src_dir + "DJI01271.jpg");
	img_names.push_back(src_dir + "DJI01272.jpg");
	img_names.push_back(src_dir + "DJI01276.jpg");
	img_names.push_back(src_dir + "DJI01277.jpg");
	img_names.push_back(src_dir + "DJI01278.jpg");
	img_names.push_back(src_dir + "DJI01279.jpg");
	img_names.push_back(src_dir + "DJI01280.jpg");
	img_names.push_back(src_dir + "DJI01281.jpg");
	img_names.push_back(src_dir + "DJI01282.jpg");
	img_names.push_back(src_dir + "DJI01283.jpg");
	img_names.push_back(src_dir + "DJI01284.jpg");
	img_names.push_back(src_dir + "DJI01285.jpg");
	return img_names;
}
vector<string> image_set2(string &src_dir)
{
	src_dir = "D:/data/my/plane/8-06_jingyuan/2/";
	vector<string> img_names;
	img_names.push_back(src_dir + "DJI01305.jpg");
	img_names.push_back(src_dir + "DJI01306.jpg");
	img_names.push_back(src_dir + "DJI01307.jpg");
	img_names.push_back(src_dir + "DJI01308.jpg");
	img_names.push_back(src_dir + "DJI01309.jpg");
	img_names.push_back(src_dir + "DJI01310.jpg");
	img_names.push_back(src_dir + "DJI01311.jpg");
	img_names.push_back(src_dir + "DJI01312.jpg");
	img_names.push_back(src_dir + "DJI01313.jpg");
	img_names.push_back(src_dir + "DJI01314.jpg");
	img_names.push_back(src_dir + "DJI01315.jpg");
	img_names.push_back(src_dir + "DJI01316.jpg");
	img_names.push_back(src_dir + "DJI01317.jpg");
	img_names.push_back(src_dir + "DJI01318.jpg");
	img_names.push_back(src_dir + "DJI01319.jpg");
	img_names.push_back(src_dir + "DJI01321.jpg");
	img_names.push_back(src_dir + "DJI01322.jpg");
	img_names.push_back(src_dir + "DJI01323.jpg");
	img_names.push_back(src_dir + "DJI01324.jpg");
	img_names.push_back(src_dir + "DJI01325.jpg");
	img_names.push_back(src_dir + "DJI01326.jpg");
	return img_names;
}

vector<string> image_set3(string &src_dir)
{
	vector<string> img_names;
	char img_name[100];
	for(int i = 1384; i <= 1539; i++)
	{
		//if(i == 1234)
			//continue;
		sprintf(img_name, "DJI0%04d.jpg", i);
		img_names.push_back(src_dir + img_name);
	}
	return img_names;
}
void aerial_stitch_slide()
{
	vector<string> img_names;// = image_set3(src_dir);
	getAllJpegs(src_dir, img_names);
	for(int i = 0; i < img_names.size(); i++)
		img_names[i] = src_dir + img_names[i];

	int num_img = img_names.size();

	//	read the images
	cout << "loading in the " << num_img << " images ... " << endl;
	//for(int i = 0; i < num_img; i++)
	//	imgs[i] = imread(img_names[i]);

	//	stitch one at each time
	vector<Mat> srcs(2);
	vector<bool> success_mask;
	srcs[0] = imread(img_names[0]);
	Mat mask0 = Mat::ones(srcs[0].size(), CV_8U) * 255;
	Mat dst_tmp, dst_mask_tmp;
	char img_name[100];
	long stitch_start_clock = clock();
	ImageFeatures feat0, dst_feat;
	vector<int> failed_index;
	for(int i = 1; i < num_img; i++)
	{
		cout << "stitch image " << i << " to the crowd" << endl;
		srcs[1] = imread(img_names[i]);
		long start_clock = clock();
		stitch_images(srcs, mask0, feat0, dst_tmp, dst_mask_tmp, dst_feat, success_mask);
		srcs[1].release();
		long end_clock = clock();
		cout << "image " << i << " costs " << end_clock-start_clock << " ms" << endl;

		if(!success_mask[1])
		{
			failed_index.push_back(i);
			cout << img_names[i] << " stitch failed" << endl;
		}

		sprintf(img_name, "debug/dst_%d.jpg", i);
		imwrite(src_dir + img_name, dst_tmp);
		sprintf(img_name, "debug/dst_mask_%d.jpg", i);
		imwrite(src_dir + img_name, dst_mask_tmp);
		mask0 = dst_mask_tmp;
		srcs[0] = dst_tmp;
		feat0 = dst_feat;

		Mat dst_feat_img;
		resize(dst_tmp, dst_feat_img, Size(), work_scale, work_scale);
		for(int j = 0; j < dst_feat.keypoints.size(); j++)
			circle(dst_feat_img, dst_feat.keypoints[j].pt, 2, Scalar(0, 0, 255, 0));
		sprintf(img_name, "debug/dst_feats_%d.jpg", i);
		imwrite(src_dir + img_name, dst_feat_img);
	}
	long stitch_end_clock = clock();

	int failed_num = failed_index.size();
	cout << num_img-failed_num << " images stitch completed, cost " << stitch_end_clock-stitch_start_clock << " ms" << endl;
	cout << failed_num << " images failed!: " << endl;
	for(int i = 0; i < failed_num; i++)
		cout << "\t" << img_names[failed_index[i]] << endl;
}

void aerial_stitch_simple()
{
	string src_dir = "D:/data/my/plane/8-06_jingyuan/";
	vector<string> img_names;
	img_names.push_back(src_dir + "DJI01270.jpg");
	img_names.push_back(src_dir + "DJI01271.jpg");
	img_names.push_back(src_dir + "DJI01272.jpg");
	img_names.push_back(src_dir + "DJI01276.jpg");
	img_names.push_back(src_dir + "DJI01277.jpg");
	img_names.push_back(src_dir + "DJI01278.jpg");

	int num_img = img_names.size();

	//	read the images
	cout << "loading in the " << num_img << " images ... " << endl;
	double work_scale = 0.2;
	vector<Mat> imgs(num_img), imgs_raw(num_img);
	char img_name[100];
	for(int i = 0; i < num_img; i++)
	{
		Mat img = imread(img_names[i]);
		imgs_raw[i] = img.clone();
		resize(img, imgs[i], Size(), work_scale, work_scale);
		sprintf(img_name, "img_%d.jpg", i);
		imwrite(src_dir + "debug/" + img_name, imgs[i]);
	}

	//	extract features
	cout << "extracting features ... " << endl;
	vector<ImageFeatures> feats(num_img);
	Ptr<FeaturesFinder> finder = new SurfFeaturesFinder();
	for(int i = 0; i < num_img; i++)
		(*finder)(imgs[i], feats[i]);
	finder->collectGarbage();

	//	feature matching
	cout << "matching the features ... " << endl;
	BestOf2NearestMatcher matcher(false, 0.3f);
	vector<MatchesInfo> matches_infos(num_img-1);
	for(int i = 1; i < num_img; i++)
	{
		matcher(feats[0], feats[i], matches_infos[i-1]);
		Mat draw_img;
		my_drawMatches2(imgs[0], imgs[i], feats[0], feats[i], matches_infos[i-1], draw_img);
		sprintf(img_name, "0_%d.jpg", i);
		imwrite(src_dir + "debug/match_" + img_name, draw_img);
	}
	matcher.collectGarbage();

	//	compute scale, theta and tx, ty
	cout << "motion estimation ... " << endl;
	vector<Point> corners(num_img);
	vector<Mat> imgs_warp_ws(num_img), masks_workscale(num_img);
	vector<Mat> Hs(num_img);
	for(int i = 0; i < num_img; i++)
	{
		if(i == 0)
			Hs[i] = cv::Mat::eye(3, 3, CV_64F);
		else
		{
			//if(compute_similar_transform(feats[0], feats[i], matches_infos[i-1], H) == -1)
			if(compute_perspective_transform(feats[0], feats[i], matches_infos[i-1], Hs[i]) == -1)
			{
				cout << "ERROR: image " << i << " stitch error" << endl;
				continue;
			}
		}
		corners[i] = my_warp_perspective(imgs[i], imgs_warp_ws[i], masks_workscale[i], Hs[i]);
	}

	//	Compensate exposure
	cout << "Compensate exposure" << endl;
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
	compensator->feed(corners, imgs_warp_ws, masks_workscale);

	//	seam
	cout << "find seam" << endl;
	vector<Mat> images_warped_f(num_img);
	for (int i = 0; i < num_img; ++i)
		imgs_warp_ws[i].convertTo(images_warped_f[i], CV_32F);
	Ptr<SeamFinder> seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	seam_finder->find(images_warped_f, corners, masks_workscale);

	//	rescale
	cout << "rescale the parameters" << endl;
	double compose_scale = 0.5, compose_work_aspect = compose_scale / work_scale;
	vector<Mat> masks_compose_scale(num_img), masks_warped_cs(num_img);
	vector<Mat> imgs_cs(num_img), imgs_warped_cs(num_img);
	vector<Size> sizes_cs(num_img);
	Mat seam_mask, dilated_mask, mask_warped;
	for(int i = 0; i < num_img; i++)
	{
		resize(imgs_raw[i], imgs_cs[i], Size(), compose_scale, compose_scale);
		//	H
		Hs[i].at<double>(0, 2) *= compose_work_aspect;
		Hs[i].at<double>(1, 2) *= compose_work_aspect;
		Hs[i].at<double>(2, 0) /= compose_work_aspect;
		Hs[i].at<double>(2, 1) /= compose_work_aspect;
		//	warp
		corners[i] = my_warp_perspective(imgs_cs[i], imgs_warped_cs[i], masks_warped_cs[i], Hs[i]);
		sizes_cs[i] = imgs_warped_cs[i].size();
		// Compensate exposure
		compensator->apply(i, corners[i], imgs_warped_cs[i], masks_warped_cs[i]);
		//	seam and mask
		dilate(masks_workscale[i], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, masks_warped_cs[i].size());
		masks_compose_scale[i] = seam_mask & masks_warped_cs[i];

		sprintf(img_name, "%d", i);
		imwrite(src_dir + "debug/warp_" + img_name + ".jpg", imgs_warped_cs[i]);
		imwrite(src_dir + "debug/mask_" + img_name + ".jpg", masks_compose_scale[i]);
	}

	//	compose
	Mat dst;
	Rect dst_roi = compose_roi(imgs_warped_cs, corners);
	cout << dst_roi << endl;

	//	blend
	cout << "blending ..." << endl;
	int blend_type = Blender::FEATHER;
	float blend_strength = 3;
	Ptr<Blender> blender = Blender::createDefault(blend_type, false);
	float blend_width = sqrt(static_cast<float>(dst_roi.area())) * blend_strength / 100.f;
	if (blend_type == Blender::MULTI_BAND)
	{
		MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
		mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
		cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
	}
	else if (blend_type == Blender::FEATHER)
	{
		FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
		fb->setSharpness(1.f/blend_width);
		cout << "Feather blender, sharpness: " << fb->sharpness() << endl;
	}
	for(int i = 0; i < num_img; i++)
		cout << corners[i] << "; " << sizes_cs[i] << endl;
	blender->prepare(corners, sizes_cs);
	cout << "\tblending prepared ..." << endl;

	for(int i = 0; i < num_img; i++)
	{
		Mat img_warped_s;
		imgs_warped_cs[i].convertTo(img_warped_s, CV_16S);
		blender->feed(img_warped_s, masks_compose_scale[i], corners[i]);
	}
	Mat result_mask;
	blender->blend(dst, result_mask);
	imwrite(src_dir + "debug/dst.jpg", dst);

	Mat dst_rough, dst_mask;
	compose_images_rough(imgs_warped_cs, masks_warped_cs, corners, dst_rough, dst_mask);
	imwrite(src_dir + "debug/dst_rough.jpg", dst_rough);
}

//	累积误差测试
void test_acc()
{
	Mat img1, img2;
	double work_scale = 0.4;
	resize(img1, img1, Size(), work_scale, work_scale);
	resize(img2, img2, Size(), work_scale, work_scale);
	ImageFeatures feat1, feat2;
	Ptr<FeaturesFinder> finder = new SurfFeaturesFinder();
	(*finder)(img1, feat1);
	(*finder)(img2, feat2);
	finder->collectGarbage();
}

//	特征提取、匹配和保存
void save_feature_match(string img_dir)
{
	//string img_dir = "data/likelou/";//"D:/data/my/plane/7-17_likelou/";
	double work_scale = 0.2;
	vector<string> img_names;
	getAllJpegs(img_dir, img_names);

	//	load images
	int num_img = img_names.size();
	cout << "loading in the " << num_img << " images ... " << endl;
	vector<Mat> imgs(num_img);
	for(int i = 0; i < num_img; i++)
	{
		cout << i << endl;
		Mat img = imread(img_dir + img_names[i]);
		//Rect rect(1000, 500, 2000, 2000);
		//img = Mat(img, rect);
		resize(img, imgs[i], Size(), work_scale, work_scale);
		img.release();
	}

	//	extract features
	cout << "extracting features ... " << endl;
	vector<ImageFeatures> feats(num_img);
	Ptr<FeaturesFinder> finder = new SurfFeaturesFinder();
	
	long long start = clock();
#pragma omp parallel for
	for(int i = 0; i < num_img; i++)
	{
		(*finder)(imgs[i], feats[i]);
		cout << "\t" << feats[i].keypoints.size() << " features for image " << img_names[i] << endl;
	}
	long long end = clock();
	cout << "extract feature " << end - start << endl;
	finder->collectGarbage();
	//for (int i = 0; i < num_img; i++)
	//	imgs[i].release();
	//imgs.clear();

	//	feature matching
	cout << "matching the features ... " << endl;
	vector<MatchesInfo> pairwise_matches(num_img * num_img);
	for(int src_idx = 0; src_idx < num_img; src_idx++)
	{
#pragma omp parallel for
		for(int dst_idx = src_idx+1; dst_idx < num_img; dst_idx++)
		{
			int match_idx = src_idx * num_img + dst_idx;
			start = clock();
			feature_match_bidirection(feats[src_idx], feats[dst_idx], pairwise_matches[match_idx]);
			end = clock();
			cout << "time cost " << end - start << endl;
			cout << "\t" << img_names[src_idx] << " " << img_names[dst_idx] << " has " << pairwise_matches[match_idx].num_inliers << " matches" << endl;
		}
	}

	//	save image names
	ofstream names_file(img_dir + "image_names.txt");
	for(int i = 0; i < num_img; i++)
		names_file << img_names[i] << endl;
	names_file.close();

	//	save features
	cout << "saving the features ... " << endl;
	ofstream feats_file(img_dir + "features.txt");
	for(int i = 0; i < num_img; i++)
	{
		feats_file << i << endl;
		feats_file << img_names[i] << endl;
		feats_file << feats[i].keypoints.size() << " ";
		for(int j = 0; j < feats[i].keypoints.size(); j++)
			feats_file << feats[i].keypoints[j].pt.x << " " << feats[i].keypoints[j].pt.y << " ";
		feats_file << endl;
		feats_file << endl;
	}
	feats_file.close();

	//	save matches
	cout << "saving the matches ... " << endl;
	for(int i = 0; i < pairwise_matches.size(); i++)
	{
		MatchesInfo match_info = pairwise_matches[i];
		int src_idx = match_info.src_img_idx;
		int dst_idx = match_info.dst_img_idx;
		//cout << src_idx << " " << dst_idx << endl;
	}
	ofstream match_file(img_dir + "matches.txt");
	for(int src_idx = 0; src_idx < num_img; src_idx++)
	{
		for(int dst_idx = src_idx+1; dst_idx < num_img; dst_idx++)
		{
			int match_idx = src_idx * num_img + dst_idx;
			MatchesInfo match_info = pairwise_matches[match_idx];
			match_file << src_idx << " " << dst_idx << endl;

			int matches_size = match_info.matches.size();
			vector<Point2f> src_pts, dst_pts;
			for(int k = 0; k < matches_size; k ++)
			{
				const DMatch& m = match_info.matches[k];
				if (match_info.inliers_mask[k])
				{
					src_pts.push_back(feats[src_idx].keypoints[m.queryIdx].pt);
					dst_pts.push_back(feats[dst_idx].keypoints[m.trainIdx].pt);
				}
			}
			int inliner_num = src_pts.size();
			//cout << "\t" << img_names[src_idx] << " " << img_names[dst_idx] << " has " << inliner_num << " matches" << endl;
			match_file << inliner_num << endl;
			for(int j = 0; j < inliner_num; j++)
				match_file << src_pts[j].x << " " << src_pts[j].y << " " << dst_pts[j].x << " " << dst_pts[j].y << endl;

			//	save match in images
			/*if(inliner_num > 0)
			{
				Mat draw_img;
				my_drawMatches(imgs[src_idx], imgs[dst_idx], src_pts, dst_pts, draw_img);
				int name_len = img_names[src_idx].length();
				imwrite(img_dir + "matches/" + img_names[src_idx].substr(0, name_len-4) + "_" + img_names[dst_idx], draw_img);
			}*/
		}
	}
	match_file.close();
}

void test_match()
{
	string img_dir = "D:/data/my/plane/7-17_likelou/";
	Mat img1 = imread(img_dir + "DJI00174.jpg");
	Mat img2 = imread(img_dir + "DJI00188.jpg");
	resize(img1, img1, Size(), work_scale, work_scale);
	resize(img2, img2, Size(), work_scale, work_scale);
	ImageFeatures feat1, feat2;
	Ptr<FeaturesFinder> finder = new SurfFeaturesFinder();
	(*finder)(img1, feat1);
	(*finder)(img2, feat2);
	MatchesInfo match_info;
	cout << "matching" << endl;
	feature_match_bidirection_raw(feat1, feat2, match_info);
	Mat draw_img;
	my_drawMatches2(img1, img2, feat1, feat2, match_info, draw_img);
	imwrite(img_dir + "match12.jpg", draw_img);
}


//	基于最小生成树的航拍图拼接
vector<Mat> aerial_stitch_mst(string img_dir, string maskFilePath)
{
	//string img_dir = "data/likelou/";//"D:/data/my/plane/7-17_likelou/";
	vector<string> img_names;
	cout << "loading image names" << endl;
	ifstream names_file(img_dir + "image_names.txt");
	string line;
	while(getline(names_file, line))
		img_names.push_back(line);
	names_file.close();

	int num_img = img_names.size();
	cout << "loading match information" << endl;
	ifstream match_file(img_dir + "matches.txt");
	vector<vector<MSTEdge>> graph_edges(num_img);
	for(int i = 0; i < num_img; i++)
		graph_edges[i].resize(num_img);
	
	while(getline(match_file, line))
	{
		int src_idx, dst_idx;
		stringstream index_ss;
		index_ss << line;
		index_ss >> src_idx >> dst_idx;

		int inliner_num;
		stringstream inline_num_ss;
		getline(match_file, line);
		inline_num_ss << line;
		inline_num_ss >> inliner_num;

		if(inliner_num >= 10 && src_idx < 200 && dst_idx < 200)
		{
			//cout << "\t" << inliner_num << endl;
			graph_edges[src_idx][dst_idx].set_value(src_idx, dst_idx, true, 1000.0 / inliner_num);
		}
		for(int j = 0; j < inliner_num; j++)
			getline(match_file, line);
	}

	cout << "building MST" << endl;
	MST mst(graph_edges);
	mst.build();
	vector<bool> is_node_used(num_img);
	vector<double> node_root_scores(num_img);
	vector<vector<int>> node_sons(num_img);
	for(int i = 0; i < num_img; i++)
	{
		is_node_used[i] = false;
		node_root_scores[i] = 0;
		for(int j = 0; j < num_img; j++)
			if(mst.edges[i][j])
			{
				node_sons[i].push_back(j);
				node_root_scores[i] += (std::min)(100.0, 1000 / (graph_edges[i][j].weight));
			}
	}

	cout << "outputing the MST" << endl;
	vector<Mat> results;
	while(true)
	{
		//	find root
		int root_idx = -1;
		double max_score = -10000;
		for(int i = 0; i < num_img; i++)
		{
			if(!is_node_used[i])
			{
				if(node_root_scores[i] > max_score)
				{
					root_idx = i;
					max_score = node_root_scores[i];
				}
			}
		}
		if(root_idx < 0)
			break;

		//	宽度优先遍历树
		vector<int> cur_node_indices, cur_sons_indices;
		cur_node_indices.push_back(root_idx);
		is_node_used[root_idx] = true;
		cout << "root: " << img_names[root_idx] << endl;
		int level = 0;

		vector<Mat> srcs;
		Mat img = imread(img_dir + img_names[root_idx]);
		resize(img, img, Size(), 0.5, 0.5);
		srcs.push_back(img);
		vector<bool> success_mask;
		Mat mask0 = Mat::ones(srcs[0].size(), CV_8U) * 255, dst_tmp, dst_mask_tmp;
		char img_name[100];
		long stitch_start_clock = clock();
		ImageFeatures feat0, dst_feat;
		vector<int> failed_index;
		while(true)
		{
			cout << "stitch images ";
			for(int i = 0; i < cur_node_indices.size(); i++)
			{
				vector<int> cur_node_sons = node_sons[cur_node_indices[i]];
				for(int j = 0; j < cur_node_sons.size(); j++)
				{
					if(!is_node_used[cur_node_sons[j]])
					{
						cur_sons_indices.push_back(cur_node_sons[j]);
						is_node_used[cur_node_sons[j]] = true;
						cout << img_names[cur_node_sons[j]] << ", ";
					}
				}
				cout << "; ";
			}
			cout << " to the crowd" << endl;
			if(cur_sons_indices.size() == 0)
				break;

			for(int i = 0; i < cur_sons_indices.size(); i++)
			{
				img = imread(img_dir + img_names[cur_sons_indices[i]]);
				resize(img, img, Size(), 0.5, 0.5);
				srcs.push_back(img);
			}
			long start_clock = clock();
			stitch_images(srcs, mask0, feat0, dst_tmp, dst_mask_tmp, dst_feat, success_mask);
			long end_clock = clock();
			cout << " costs " << end_clock-start_clock << " ms" << endl;
			for(int i = 1; i < srcs.size(); i++)
			{
				if(!success_mask[i])
				{
					failed_index.push_back(cur_sons_indices[i-1]);
					cout << img_names[cur_sons_indices[i-1]] << " stitch failed" << endl;
				}
			}
			for(int i = 0; i < srcs.size(); i++)
				srcs[i].release();
			srcs.clear();

			sprintf(img_name, "debug/dst_%d.jpg", level);
			imwrite(src_dir + img_name, dst_tmp);
			sprintf(img_name, "debug/dst_mask_%d.jpg", level);
			imwrite(src_dir + img_name, dst_mask_tmp);
			mask0 = dst_mask_tmp;
			srcs.push_back(dst_tmp.clone());
			feat0 = dst_feat;

			//	copy sons as nodes of next iteration
			cur_node_indices.resize(cur_sons_indices.size());
			for(int i = 0; i < cur_node_indices.size(); i++)
				cur_node_indices[i] = cur_sons_indices[i];
			cur_sons_indices.clear();
			level++;
		}
		cout << endl;
		results.push_back(dst_tmp.clone());

		long stitch_end_clock = clock();

		int failed_num = failed_index.size();
		cout << num_img-failed_num << " images stitch completed, cost " << stitch_end_clock-stitch_start_clock << " ms" << endl;
		cout << failed_num << " images failed!: " << endl;
		for(int i = 0; i < failed_num; i++)
			cout << "\t" << img_names[failed_index[i]] << endl;

		break;
	}

	return results;
}

int main()
{
	//aerial_stitch();
	//aerial_stitch_slide();

	//	先做图像两两匹配，确定邻接图模型，并保存匹配结果。如果已经保存过匹配结果，则无需执行此步
	cout << "请输入待拼接的图像数据集目录:" << endl;
	cin >> src_dir;
	src_dir += '\\';
	cout << src_dir << endl;
	string maskFilePath = "F:\\data.txt";
	mkdir((src_dir + "debug").c_str());
	save_feature_match(src_dir);

	//	然后进行拼接。
	
	vector<Mat> results = aerial_stitch_mst(src_dir, maskFilePath);

	//test_match();
	system("pause");
	return 0;
}