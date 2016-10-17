
#include "util.h"

using namespace std;
using namespace cv;
using namespace cv::detail;


//	argmin(||Ax - b||^2)
//	A:	n*m
//	x:	m*1
//	b:	n*1
void least_square(Mat A, Mat b, Mat &x)
{
	Mat A_t = A.t();
	x = (A_t * A).inv() * A_t * b;
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

void my_drawMatches2(Mat img1, Mat img2, ImageFeatures feat1, ImageFeatures feat2, MatchesInfo matches_info, Mat &draw_img)
{
	// 特征点对处理
	vector<Point2f> src_pts, dst_pts;
	int matches_size = matches_info.matches.size();
	int inliner_nums = matches_info.num_inliers;

	int match_inline_num = 0;
	for(int k = 0; k < matches_size; k ++)
	{
		const DMatch& m = matches_info.matches[k];
		Point2f p1 = feat1.keypoints[m.queryIdx].pt;
		Point2f p2 = feat2.keypoints[m.trainIdx].pt;
		if (matches_info.inliers_mask[k])
		{
			src_pts.push_back(p1);
			dst_pts.push_back(p2);
			match_inline_num ++;
		}
	}
	my_drawMatches(img1, img2, src_pts, dst_pts, draw_img);
}

void getAllJpegs(string dirpath, std::vector <string> &filenames)
{
	string dir_spec = dirpath + "/*.png";
	WIN32_FIND_DATAA f;
	HANDLE h = FindFirstFileA(dir_spec.c_str() , &f);
	filenames.clear();
	if(h != INVALID_HANDLE_VALUE)
	{
		do
		{
			filenames.push_back( string(f.cFileName) );
		}while(FindNextFileA(h, &f));
	}
	FindClose(h);

	dir_spec = dirpath + "/*.jpg";
	h = FindFirstFileA(dir_spec.c_str() , &f);
	filenames.clear();
	if(h != INVALID_HANDLE_VALUE)
	{
		do
		{
			filenames.push_back( string(f.cFileName) );
		}while(FindNextFileA(h, &f));
	}
	FindClose(h);
}
