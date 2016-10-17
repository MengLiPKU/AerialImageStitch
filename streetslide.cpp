
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::detail;

void my_drawMatches(Mat img1, Mat img2, vector<Point> src_pts, vector<Point> dst_pts, Mat &draw_img)
{
	int row1 = img1.rows;
	int col1 = img1.cols;
	int row2 = img2.rows;
	int col2 = img2.cols;
	draw_img.create(((row1 > row2) ? row1 : row2), col1 + col2, CV_8UC3);
	img1.copyTo(draw_img(Rect(0, 0, col1, row1)));
	img2.copyTo(draw_img(Rect(col1, 0, col2, row2)));
	int num_pts = src_pts.size();
	int radius = 2;
	Scalar color_pt = Scalar(255, 0, 0, 0);
	Scalar color_line = Scalar(0, 0, 255, 0);
	char num_str[10];
	for(int i = 0 ; i < num_pts; i++)
	{
		circle(draw_img, src_pts[i], radius, color_pt, radius);
		circle(draw_img, dst_pts[i]+Point(col1, 0), radius, color_pt, radius);
		line(draw_img, src_pts[i], dst_pts[i]+Point(col1, 0), color_line);
		sprintf(num_str, "%d", i);
		putText(draw_img, string(num_str), src_pts[i], CV_FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255, 0), 1);
		putText(draw_img, string(num_str), dst_pts[i]+Point(col1, 0), CV_FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255, 0), 1);
	}
}

void my_drawMatches(Mat img1, Mat img2, vector<Point2f> src_pts, vector<Point2f> dst_pts, Mat &draw_img)
{
	int row1 = img1.rows;
	int col1 = img1.cols;
	int row2 = img2.rows;
	int col2 = img2.cols;
	draw_img.create(((row1 > row2) ? row1 : row2), col1 + col2, CV_8UC3);
	img1.copyTo(draw_img(Rect(0, 0, col1, row1)));
	img2.copyTo(draw_img(Rect(col1, 0, col2, row2)));
	int num_pts = src_pts.size();
	int radius = 2;
	Scalar color_pt = Scalar(255, 0, 0, 0);
	Scalar color_line = Scalar(0, 0, 255, 0);
	char num_str[10];
	for(int i = 0 ; i < num_pts; i++)
	{
		circle(draw_img, Point(src_pts[i]), radius, color_pt, radius);
		circle(draw_img, Point(dst_pts[i])+Point(col1, 0), radius, color_pt, radius);
		line(draw_img, Point(src_pts[i]), Point(dst_pts[i])+Point(col1, 0), color_line);
		sprintf(num_str, "%d", i);
		putText(draw_img, string(num_str), Point(src_pts[i]), CV_FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255, 0), 1);
		putText(draw_img, string(num_str), Point(dst_pts[i])+Point(col1, 0), CV_FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255, 0), 1);
	}
}

void my_drawMatches(Mat img1, Mat img2, ImageFeatures feat1, ImageFeatures feat2, MatchesInfo matches_info, 
	Rect roi1, Rect roi2, Mat &draw_img)
{
	// 特征点对处理
	vector<Point> src_pts, dst_pts;
	int matches_size = matches_info.matches.size();
	int inliner_nums = matches_info.num_inliers;

	int match_inline_num = 0;
	for(int k = 0; k < matches_size; k ++)
	{
		const DMatch& m = matches_info.matches[k];
		Point p1 = feat1.keypoints[m.queryIdx].pt;
		Point p2 = feat2.keypoints[m.trainIdx].pt;
		if (matches_info.inliers_mask[k])
		{
			src_pts.push_back(p1 + roi1.tl());
			dst_pts.push_back(p2 + roi2.tl());
			match_inline_num ++;
		}
	}
	my_drawMatches(img1, img2, src_pts, dst_pts, draw_img);
}

void save_matrix(string filename, Mat m)
{
	ofstream mfile(filename.c_str());
	mfile << m.rows << " " << m.cols << endl;
	for(int i = 0; i < m.rows; i++)
	{
		for(int j = 0; j < m.cols; j++)
		{
			if(m.type() == CV_32F)
				mfile << m.at<float>(i, j) << " ";
			else
				mfile << m.at<double>(i, j) << " ";
		}
		mfile << endl;
	}
}

void align_images()
{
	string dir_path = "D:\\data\\my\\north\\lijiao\\";
	Mat img1 = imread(dir_path + "IMG_1682.jpg");
	Mat img2 = imread(dir_path + "IMG_1683.jpg");

	//	先全局做一次配准，计算出H，并让H(1, 3)=0
	double scale1 = 0.25;
	resize(img1, img1, Size(), scale1, scale1);
	resize(img2, img2, Size(), scale1, scale1);
	ImageFeatures feat1;
	ImageFeatures feat2;
	Ptr<FeaturesFinder> finder = new SurfFeaturesFinder();
	(*finder)(img1, feat1);
	(*finder)(img2, feat2);
	MatchesInfo matches_info;
	BestOf2NearestMatcher matcher(false, 0.3f);
	matcher(feat1, feat2, matches_info);
	finder->collectGarbage();
	int matches_size = matches_info.matches.size();
	vector<Point2f> src_pts, dst_pts;
	for(int k = 0; k < matches_size; k ++)
	{
		const DMatch& m = matches_info.matches[k];
		Point p1 = feat1.keypoints[m.queryIdx].pt;
		Point p2 = feat2.keypoints[m.trainIdx].pt;
		if (matches_info.inliers_mask[k])
		{
			src_pts.push_back(p1);
			dst_pts.push_back(p2);
		}
	}
	matcher.collectGarbage();
	Mat H = findHomography(src_pts, dst_pts, CV_RANSAC);
	if(H.type() == CV_32F)
		H.at<float>(0, 2) = 0;
	else
		H.at<double>(0, 2) = 0;
	cout << H << endl;
	cout << H.type() << endl;
	Mat img2_warp;
	warpPerspective(img2, img2_warp, H, Size());

	int width = img1.cols;
	int height = img1.rows;
	int col1 = width / 2;
	int row_start = height / 4;
	int row_end = height / 2;
	int align_width = width / 10;

	int x1 = col1 - align_width/2;
	int y1 = row_start;
	int x2 = col1 - align_width* 5 / 2;
	int y2 = row_start;
	Rect roi1 = Rect(x1, y1, align_width, row_end-row_start);
	Rect roi2 = Rect(x2, y2, align_width * 3, row_end-row_start);
	//(*finder)(img1(roi1), feat1);
	//(*finder)(img2(roi2), feat2);
	//(*finder)(img1, feat1);
	//(*finder)(img2, feat2);

	//	匹配

	Mat matches_img;
	my_drawMatches(img1, img2, feat1, feat2, matches_info, roi1, roi2, matches_img);

	// 保存
	char img_name[100];
	sprintf(img_name, "%d_%d.jpg", 1682, 1683);
	imwrite(dir_path + img_name, matches_img);
	imwrite(dir_path + "1683_warp.jpg", img2_warp);
}

void preprocess_images()
{
	//	读取图片
	string dir_path = "D:\\data\\my\\north\\lijiao\\";
	string warped_path = "D:\\data\\my\\north\\lijiao\\warped\\";
	vector<string> img_names;
	char img_name[100];
	int start_idx = 1680, end_idx = 1690;
	for(int i = start_idx; i <= end_idx; i++)
	{
		sprintf(img_name, "IMG_%d.jpg", i);
		img_names.push_back(dir_path + img_name);
	}
	int num_imgs = img_names.size();
	vector<Mat> images(num_imgs);
	double scale = 0.25;
	Mat img;
	for(int i = 0; i < num_imgs; i++)
	{
		img = imread(img_names[i]);
		resize(img, images[i], Size(), scale, scale);
	}

	//	特征提取
	vector<ImageFeatures> feats(num_imgs);
	Ptr<FeaturesFinder> finder = new SurfFeaturesFinder();
	for(int i = 0; i < num_imgs; i++)
	{
		cout << "feature extraction for image " << i << endl;
		(*finder)(images[i], feats[i]);
	}
	
	//	特征匹配
	BestOf2NearestMatcher matcher(false, 0.3f);
	vector<MatchesInfo> matches(num_imgs - 1);
	Mat H_acc = Mat::eye(3, 3, CV_64F);
	int img_height = images[0].rows;
	int img_width = images[0].cols;
	int pts_y_start = img_height * 0.25;
	int pts_y_end = img_height * 5 / 8;
	for(int i = 0; i < (num_imgs-1); i++)
	{
		cout << "matching for image " << i << " and image " << i+1 << endl;
		matcher(feats[i], feats[i+1], matches[i]);
		MatchesInfo matches_info = matches[i];
		int matches_size = matches_info.matches.size();
		vector<Point2f> src_pts, dst_pts;
		for(int k = 0; k < matches_size; k ++)
		{
			const DMatch& m = matches_info.matches[k];
			Point p1 = feats[i].keypoints[m.queryIdx].pt;
			Point p2 = feats[i+1].keypoints[m.trainIdx].pt;
			if (matches_info.inliers_mask[k])
			{
				if((p1.y <= pts_y_end) && (p1.y >= pts_y_start) && (p2.y <= pts_y_end) && (p2.y >= pts_y_start))
				{
					src_pts.push_back(p2);
					dst_pts.push_back(p1);
				}
			}
		}
		Mat img_matches;
		my_drawMatches(images[i], images[i+1], dst_pts, src_pts, img_matches);
		sprintf(img_name, "%d_%d.jpg", i, i+1);
		imwrite(warped_path + img_name, img_matches);

		Mat H = findHomography(src_pts, dst_pts, CV_RANSAC);
		cout << "H: " << H << endl;

		H_acc = H_acc * H;
		cout << "H_acc: " << H_acc << endl;
		//	有视差版本
		Mat img_warp;
		warpPerspective(images[i+1], img_warp, H_acc, Size());//, INTER_LINEAR |  WARP_INVERSE_MAP);
		sprintf(img_name, "%d.jpg", i+1);
		imwrite(warped_path + "disparity\\" + img_name, img_warp);

		//	无视差版本
		Mat H_tmp = H_acc;
		if(H_tmp.type() == CV_32F)
			H_tmp.at<float>(0, 2) = 0;
		else
			H_tmp.at<double>(0, 2) = 0;
		warpPerspective(images[i+1], img_warp, H_tmp, Size());
		sprintf(img_name, "%d.jpg", i+1);
		imwrite(warped_path + img_name, img_warp);
	}
	imwrite(warped_path + "disparity\\0.jpg", images[0]);
	imwrite(warped_path + "0.jpg", images[0]);

	finder->collectGarbage();
	matcher.collectGarbage();
}

void my_warp_perspective(Mat src, Mat &dst, Mat H)
{
	int src_rows = src.rows;
	int src_cols = src.cols;
	Mat_<double> H_ = H, src_p(3, 1), dst_p(3, 1), H_inv = H.inv();
	src_p(0, 0) = src_cols;
	src_p(1, 0) = 0;
	src_p(2, 0) = 1;
	dst_p = H_ * src_p;
	double x1 = dst_p(0, 0) / dst_p(2, 0);

	src_p(1, 0) = src_rows-1;
	dst_p = H_ * src_p;
	double x2 = dst_p(0, 0) / dst_p(2, 0);
	//double x1 = (H_(0, 0) * src_cols + H_(0, 2)) / (H_(2, 0) * src_cols + H_(2, 2));
	//double x2 = (H_(0, 0) * src_cols + H_(0, 1) * src_rows + H_(0, 2)) / (H_(2, 0) * src_cols + H_(2, 1) * src_rows + H_(2, 2));
	int dst_width = cvCeil(max(x1, x2));
	int dst_height = src_rows;

	cout << dst_width << ", " << dst_height << endl;
	Mat xmap(dst_height, dst_width, CV_32F);
	Mat ymap(dst_height, dst_width, CV_32F);
	float *xmap_rev_ptr = xmap.ptr<float>(0);
	float *ymap_rev_ptr = ymap.ptr<float>(0);
	dst_p(2, 0) = 1;
	for(int y = 0; y < dst_height; y++)
	{
		for(int x = 0; x < dst_width; x++)
		{
			int idx = y * dst_width + x;
			dst_p(0, 0) = x;
			dst_p(1, 0) = y;
			src_p = H_inv * dst_p;
			xmap_rev_ptr[idx] = src_p(0, 0) / src_p(2, 0);
			ymap_rev_ptr[idx] = src_p(1, 0) / src_p(2, 0);
		}
	}
	remap(src, dst, xmap, ymap, INTER_LINEAR);
}

void preprocess_images2()
{
	//	读取图片
	string dir_path = "D:\\data\\my\\north\\sr\\";//"D:\\data\\my\\north\\lijiao\\";
	string warped_path = "D:\\data\\my\\north\\sr\\warp\\";
	vector<string> img_names;
	char img_name[100];
	/*
	int start_idx = 1, end_idx = 30;
	for(int i = start_idx; i <= end_idx; i++)
	{
		//sprintf(img_name, "IMG_%d.jpg", i);
		sprintf(img_name, "(%d).jpg", i);
		img_names.push_back(dir_path + img_name);
	}*/
	img_names.push_back(dir_path + "fv1.JPG");
	img_names.push_back(dir_path + "IMG_0574.JPG");
	img_names.push_back(dir_path + "IMG_0575.JPG");
	img_names.push_back(dir_path + "IMG_0576.JPG");


	int num_imgs = img_names.size();
	vector<Mat> images(num_imgs);
	double scale = 0.25;
	Mat img;
	for(int i = 0; i < num_imgs; i++)
	{
		images[i] = imread(img_names[i]);
		//resize(img, images[i], Size(), scale, scale);
	}

	//	特征提取
	Ptr<FeaturesFinder> finder = new SurfFeaturesFinder();

	//	特征匹配
	BestOf2NearestMatcher matcher(false, 0.3f);
	vector<MatchesInfo> matches(num_imgs - 1);
	int img_height = images[0].rows;
	int img_width = images[0].cols;
	int pts_y_start = img_height * 0;
	int pts_y_end = img_height * 6 / 8;
	Mat img_last_warp = images[0];
	bool is_disparity = true;
	for(int i = 0; i < (num_imgs-1); i++)
	{
		cout << "matching for image " << i << " and image " << i+1 << endl;

		ImageFeatures feat1;
		ImageFeatures feat2;
		(*finder)(img_last_warp, feat1);
		(*finder)(images[i+1], feat2);
		MatchesInfo matches_info;
		BestOf2NearestMatcher matcher(false, 0.3f);
		matcher(feat1, feat2, matches_info);
		int matches_size = matches_info.matches.size();
		vector<Point2f> src_pts, dst_pts;
		for(int k = 0; k < matches_size; k ++)
		{
			const DMatch& m = matches_info.matches[k];
			Point p1 = feat1.keypoints[m.queryIdx].pt;
			Point p2 = feat2.keypoints[m.trainIdx].pt;
			if (matches_info.inliers_mask[k])
			{
				if((p1.y <= pts_y_end) && (p1.y >= pts_y_start) && (p2.y <= pts_y_end) && (p2.y >= pts_y_start))
				{
					src_pts.push_back(p2);
					dst_pts.push_back(p1);
				}
			}
		}
		Mat H = findHomography(src_pts, dst_pts, CV_RANSAC);
		cout << "H: " << H << endl;

		//	有视差版本
		Mat img_warp;
		if(is_disparity)
		{
			Mat img1(images[i+1].rows, images[i+1].cols, CV_8UC3);
			images[i+1].copyTo(img1(Rect(0, 0, images[i+1].cols, images[i+1].rows)));
			warpPerspective(img1, img_warp, H, Size(), INTER_LINEAR );//, INTER_LINEAR |  WARP_INVERSE_MAP);
			//my_warp_perspective(images[i+1], img_warp, H);
			sprintf(img_name, "fv1_%d.jpg", i+1);
			imwrite(warped_path + img_name, img_warp);
			sprintf(img_name, "fv1_H_%d.txt", i+1);
			save_matrix(warped_path + img_name, H);
		}
		else
		{
			if(H.type() == CV_32F)
				H.at<float>(0, 2) = 0;
			else
				H.at<double>(0, 2) = 0;
			warpPerspective(images[i+1], img_warp, H, Size(), INTER_LINEAR );//, INTER_LINEAR |  WARP_INVERSE_MAP);
			sprintf(img_name, "%d.jpg", i+1);
			imwrite(warped_path + img_name, img_warp);
		}

		img_last_warp = img_warp;
	}
	if(is_disparity)
	{
		Mat img1(1200, 1600, CV_8UC3);
		images[0].copyTo(img1(Rect(0, 0, 1600, 1200)));
		imwrite(warped_path + "fv_0.jpg", img1);
	}
	else
		imwrite(warped_path + "0.jpg", images[0]);

	finder->collectGarbage();
	matcher.collectGarbage();
}

void slide_stitch()
{
	string warped_path = "D:\\data\\my\\north\\NorthSquare\\slide\\1\\";//"D:\\data\\my\\north\\lijiao\\warped\\disparity\\";
	int num = 30;
	int start_col = 480;
	int end_col = 1650;
	double slide_width = ((double)(end_col - start_col)) / num;
	char img_name[100];
	Mat res(640, 2000, CV_8UC3);
	Rect src_roi, dst_roi;
	src_roi.x = dst_roi.x = 0;
	src_roi.y = dst_roi.y = 0;
	src_roi.height = dst_roi.height = 640;
	Mat last_img;
	for(int i = 0; i < num; i++)
	{
		sprintf(img_name, "%d.jpg", i);
		Mat img = imread(warped_path + img_name);
		int slide_start_col = cvFloor(480 + slide_width * i);
		dst_roi.x = slide_start_col;
		if(i == 0)
		{
			src_roi.x = dst_roi.x = 0;
			src_roi.width = dst_roi.width = 480 + cvCeil(slide_width);
		}
		else
		{
			if(i == (num-1))
				src_roi.width = dst_roi.width = 2000 - 1 - dst_roi.x;
			else
				src_roi.width = dst_roi.width = cvCeil(slide_width);

			Mat last_align = last_img(Rect(slide_start_col - 30, 100, 30, 320));
			double ssd_min = -1;
			for(int c = slide_start_col - 60; c < slide_start_col; c++)
			{
				double ssd = norm(last_align, img(Rect(c, 100, 30, 320)));
				if(ssd_min == -1 || ssd < ssd_min)
				{
					ssd_min = ssd;
					src_roi.x = c;
				}
			}
		}

		cout << src_roi.x << ", " << dst_roi.x << ". " << src_roi.width << endl;
		img(src_roi).copyTo(res(dst_roi));
		last_img = img;
	}
	imwrite(warped_path + "res.jpg", res);
}

void front_view()
{
	string dir_path = "D:\\data\\my\\north\\sr\\";
	string warped_path = "D:\\data\\my\\north\\sr\\warp\\";
	vector<string> img_names;
	char img_name[100];
	img_names.push_back(dir_path + "fv1.JPG");
	img_names.push_back(dir_path + "IMG_0574.JPG");
	img_names.push_back(dir_path + "IMG_0575.JPG");
	img_names.push_back(dir_path + "IMG_0576.JPG");

	int num_imgs = img_names.size();
	vector<Mat> images(num_imgs);
	double scale = 0.25;
	Mat img;
	for(int i = 0; i < num_imgs; i++)
	{
		images[i] = imread(img_names[i]);
		//resize(img, images[i], Size(), scale, scale);
	}

	//	特征提取
	vector<ImageFeatures> feats(num_imgs);
	Ptr<FeaturesFinder> finder = new SurfFeaturesFinder();
	for(int i = 0; i < num_imgs; i++)
	{
		cout << "feature extraction for image " << i << endl;
		(*finder)(images[i], feats[i]);
	}

	//	特征匹配
	BestOf2NearestMatcher matcher(false, 0.3f);
	int img_height = images[0].rows;
	int img_width = images[0].cols;
	int pts_y_start = img_height * 0;
	int pts_y_end = img_height;
	for(int i = 1; i < num_imgs; i++)
	{
		cout << "matching for image 0 and image " << i << endl;

		MatchesInfo matches_info;
		BestOf2NearestMatcher matcher(false, 0.3f);
		matcher(feats[0], feats[i], matches_info);
		int matches_size = matches_info.matches.size();
		vector<Point2f> src_pts, dst_pts;
		for(int k = 0; k < matches_size; k ++)
		{
			const DMatch& m = matches_info.matches[k];
			Point p1 = feats[0].keypoints[m.queryIdx].pt;
			Point p2 = feats[i].keypoints[m.trainIdx].pt;
			if (matches_info.inliers_mask[k])
			{
				if((p1.y <= pts_y_end) && (p1.y >= pts_y_start) && (p2.y <= pts_y_end) && (p2.y >= pts_y_start))
				{
					src_pts.push_back(p2);
					dst_pts.push_back(p1);
				}
			}
		}
		cout << src_pts.size() << ", " << dst_pts.size() << endl;
		Mat H = findHomography(src_pts, dst_pts, CV_RANSAC);
		cout << "H: " << H << endl;

		Mat img_warp;
		Mat img1(images[i].rows, 1600, CV_8UC3);
		images[i].copyTo(img1(Rect(0, 0, images[i].cols, images[i].rows)));
		warpPerspective(img1, img_warp, H, Size(), INTER_LINEAR );//, INTER_LINEAR |  WARP_INVERSE_MAP);
		//my_warp_perspective(images[i+1], img_warp, H);
		sprintf(img_name, "fv1_%d.jpg", i);
		imwrite(warped_path + img_name, img_warp);
	}

	finder->collectGarbage();
	matcher.collectGarbage();

}
