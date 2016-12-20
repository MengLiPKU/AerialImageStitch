
#include "warp.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

//	计算相似变换
static void similar_transform_(vector<Point2f> src_pts, vector<Point2f> dst_pts, Mat &H)
{
	int n = src_pts.size();
	Mat A(2 * n, 4, CV_64F);
	Mat b(2 * n, 1, CV_64F);
	Mat x;
	for(int i = 0; i < n; i++)
	{
		A.at<double>(2 * i, 0) = src_pts[i].x;
		A.at<double>(2 * i, 1) = -src_pts[i].y;
		A.at<double>(2 * i, 2) = 1;
		A.at<double>(2 * i, 3) = 0;
		A.at<double>(2 * i + 1, 0) = src_pts[i].y;
		A.at<double>(2 * i + 1, 1) = src_pts[i].x;
		A.at<double>(2 * i + 1, 2) = 0;
		A.at<double>(2 * i + 1, 3) = 1;
		b.at<double>(2 * i, 0) = dst_pts[i].x;
		b.at<double>(2 * i + 1, 0) = dst_pts[i].y;
	}
	least_square(A, b, x);
	if(H.empty())
		H = Mat::zeros(3, 3, CV_64F);
	H.at<double>(0, 0) = x.at<double>(0, 0);	// a
	H.at<double>(0, 1) = -x.at<double>(1, 0);	// -b
	H.at<double>(0, 2) = x.at<double>(2, 0);	// tx
	H.at<double>(1, 0) = x.at<double>(1, 0);	// b
	H.at<double>(1, 1) = x.at<double>(0, 0);	// a
	H.at<double>(1, 2) = x.at<double>(3, 0);	// ty
	H.at<double>(2, 2) = 1;
}

// 计算仿射变换
static void affine_transform_(vector<Point2f> src_pts, vector<Point2f> dst_pts, Mat &H)
{
	int n = src_pts.size();
	Mat A = Mat::zeros(2 * n, 6, CV_64F);
	Mat b(2 * n, 1, CV_64F);
	Mat x;
	for(int i = 0; i < n; i++)
	{
		A.at<double>(2 * i, 0) = src_pts[i].x;
		A.at<double>(2 * i, 1) = src_pts[i].y;
		A.at<double>(2 * i, 2) = 1;
		A.at<double>(2 * i + 1, 3) = src_pts[i].x;
		A.at<double>(2 * i + 1, 4) = src_pts[i].y;
		A.at<double>(2 * i + 1, 5) = 1;
		b.at<double>(2 * i, 0) = dst_pts[i].x;
		b.at<double>(2 * i + 1, 0) = dst_pts[i].y;
	}
	least_square(A, b, x);
	if(H.empty())
		H = Mat::zeros(3, 3, CV_64F);
	H.at<double>(0, 0) = x.at<double>(0, 0);	// a
	H.at<double>(0, 1) = x.at<double>(1, 0);	// b
	H.at<double>(0, 2) = x.at<double>(2, 0);	// tx
	H.at<double>(1, 0) = x.at<double>(3, 0);	// c
	H.at<double>(1, 1) = x.at<double>(4, 0);	// d
	H.at<double>(1, 2) = x.at<double>(5, 0);	// ty
	H.at<double>(2, 2) = 1;
}

// 计算内点，返回内点数量，填充数组inliner_mask
int get_inliners(vector<Point2f> src_pts, vector<Point2f> dst_pts, Mat H, double thresh_error, vector<bool> &inliner_mask)
{
	int total_num = src_pts.size();
	inliner_mask.resize(total_num);
	int inliner_num = 0;
	for(int i = 0; i < total_num; i++)
	{
		Point2f warp_p = warp_point(src_pts[i], H);
		Point2f delta_p = warp_p - dst_pts[i];
		double err = delta_p.x * delta_p.x + delta_p.y * delta_p.y;
		if(err <= thresh_error)
		{
			inliner_mask[i] = true;
			inliner_num++;
		}
		else
			inliner_mask[i] = false;
	}
	return inliner_num;
}

//	计算变换矩阵
//	输入：
//		H：单应性矩阵
//		trans_type：AS_SIMILAR 或 AS_AFFINE
//		thresh_error：误差阈值
//	返回：
//		-1：点对数量不够
//		0：成功
int compute_transform(ImageFeatures feat1, ImageFeatures feat2, MatchesInfo matches_info, Mat &H, int trans_type, double thresh_error)
{
	int matches_size = matches_info.matches.size();
	vector<Point2f> src_pts, dst_pts;
	for(int k = 0; k < matches_size; k ++)
	{
		const DMatch& m = matches_info.matches[k];
		Point2f p1 = feat1.keypoints[m.queryIdx].pt;
		Point2f p2 = feat2.keypoints[m.trainIdx].pt;
		if (matches_info.inliers_mask[k])
		{
			src_pts.push_back(p2);
			dst_pts.push_back(p1);
		}
	}
	int total_num = src_pts.size();
	if(total_num <= 5)
	{
		cout << total_num << "inliner matches are no enough!" << endl;
		return -1;
	}

	//	ransac
	int max_inliner_num = 0;
	for(int iter = 0; iter < 100; iter++)
	{
		int m = 6;
		vector<int> indices(m);
		srand(iter * 13);
		for(int i = 0; i < m; )
		{
			int r = random(total_num);
			bool is_r_exist = false;
			for(int j = 0; j < i; j++)
				if(indices[j] == r)
					is_r_exist = true;
			if(!is_r_exist)
				indices[i++] = r;
		}
		vector<Point2f> src_pts_epoch, dst_pts_epoch;
		for(int i = 0; i < m; i++)
		{
			src_pts_epoch.push_back(src_pts[indices[i]]);
			dst_pts_epoch.push_back(dst_pts[indices[i]]);
		}
		Mat H_tmp;
		if(trans_type == AS_SIMILAR)
			similar_transform_(src_pts_epoch, dst_pts_epoch, H_tmp);
		else if(trans_type == AS_AFFINE)
			affine_transform_(src_pts_epoch, dst_pts_epoch, H_tmp);

		//	get the inliners
		vector<Point2f> src_in_pts, dst_in_pts;
		vector<bool> inliner_mask;
		int inliner_num = get_inliners(src_pts, dst_pts, H_tmp, thresh_error, inliner_mask);
		for(int i = 0; i < total_num; i++)
		{
			if(inliner_mask[i])
			{
				src_in_pts.push_back(src_pts[i]);
				dst_in_pts.push_back(dst_pts[i]);
			}
		}

		if(inliner_num > max_inliner_num && inliner_num > 3)
		{
			if(trans_type == AS_SIMILAR)
				similar_transform_(src_in_pts, dst_in_pts, H);
			else if(trans_type == AS_AFFINE)
				affine_transform_(src_in_pts, dst_in_pts, H);
			max_inliner_num = inliner_num;
		}
		if(inliner_num >= 0.9*total_num)
			break;
	}

	cout << "\t" << total_num << " match inliners initially, " << max_inliner_num << " match inliners finally" << endl;
	if(max_inliner_num <= 3)
	{
		cout << max_inliner_num << "inliner matches are no enough!" << endl;
		return -1;
	}
	if(max_inliner_num <= 6)
		cout << "WARNING: " << max_inliner_num << " inliners may not be enough" << endl;

	return 0;
}

//	透视变换
int compute_perspective_transform(ImageFeatures feat1, ImageFeatures feat2, MatchesInfo matches_info, Mat &H, double thresh_error)
{
	int matches_size = matches_info.matches.size();
	vector<Point2f> src_pts, dst_pts;
	for(int k = 0; k < matches_size; k ++)
	{
		const DMatch& m = matches_info.matches[k];
		Point p1 = feat1.keypoints[m.queryIdx].pt;
		Point p2 = feat2.keypoints[m.trainIdx].pt;
		if (matches_info.inliers_mask[k])
		{
			src_pts.push_back(p2);
			dst_pts.push_back(p1);
		}
	}
	Mat H_ = findHomography(src_pts, dst_pts, CV_RANSAC);
	H_.convertTo(H, CV_64F);
	return 0;
}

//	一个点做变换
Point2f warp_point(Point2f p, Mat H)
{
	Mat_<double> H_ = H, p_(3, 1), dst_p(3, 1);
	p_(0, 0) = p.x;
	p_(1, 0) = p.y;
	p_(2, 0) = 1;
	dst_p = H_ * p_;
	return Point2f(dst_p(0, 0) / dst_p(2, 0), dst_p(1, 0) / dst_p(2, 0));
}

//	一个矩形做变换
Rect warp_area(Mat H, Rect r)
{
	int x1, x2, y1, y2;
	x1 = r.x;
	x2 = x1 + r.width;
	y1 = r.y;
	y2 = y1 + r.height;
	Point2d tl = Point2d(warp_point(Point2f(x1, y1), H));
	Point2d bl = Point2d(warp_point(Point2f(x1, y2), H));
	Point2d tr = Point2d(warp_point(Point2f(x2, y1), H));
	Point2d br = Point2d(warp_point(Point2f(x2, y2), H));
	Rect warp_r;
	warp_r.x = cvFloor((std::min)( (std::min)(tl.x, bl.x), (std::min)(tr.x, br.x) ));
	warp_r.y = cvFloor((std::min)( (std::min)(tl.y, bl.y), (std::min)(tr.y, br.y) ));
	warp_r.width = cvCeil((std::max)( (std::max)(tl.x, bl.x), (std::max)(tr.x, br.x) )) - warp_r.x;
	warp_r.height = cvCeil((std::max)( (std::max)(tl.y, bl.y), (std::max)(tr.y, br.y) )) - warp_r.y;
	return warp_r;
}

//	opencv自带的warp_perspective有问题，会丢掉所有负象限的图像内容
//	自己实现的，效率较低
Point my_warp_perspective(Mat src, Mat &dst, Mat &mask, Mat H)
{
	int src_rows = src.rows;
	int src_cols = src.cols;
	Rect warped_rect = warp_area(H, Rect(0, 0, src_cols, src_rows));
	int dst_width = warped_rect.width;
	int dst_height = warped_rect.height;
	Point corner = warped_rect.tl();

	//cout << "\t" << corner.x << ", " << corner.y << ": " << dst_width << ", " << dst_height << endl;
	Mat xmap(dst_height, dst_width, CV_32F);
	Mat ymap(dst_height, dst_width, CV_32F);
	float *xmap_rev_ptr = xmap.ptr<float>(0);
	float *ymap_rev_ptr = ymap.ptr<float>(0);

	Mat_<double> H_inv = H.inv();
	//cout << H_inv << endl;
	for(int y = 0; y < dst_height; y++)
	{
		for(int x = 0; x < dst_width; x++)
		{
			Mat_<double> dst_p(3, 1), src_p;
			int idx = y * dst_width + x;
			dst_p(0, 0) = x + corner.x;
			dst_p(1, 0) = y + corner.y;
			dst_p(2, 0) = 1;
			src_p = H_inv * dst_p;
			xmap_rev_ptr[idx] = src_p(0, 0) / src_p(2, 0);
			ymap_rev_ptr[idx] = src_p(1, 0) / src_p(2, 0);
		}
	}
	remap(src, dst, xmap, ymap, INTER_LINEAR);
	Mat src_mask = cv::Mat::ones(src_rows, src_cols, CV_8U) * 255;
	remap(src_mask, mask, xmap, ymap, INTER_LINEAR);
	return corner;
}