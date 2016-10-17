#ifndef __ZFD_AS_UTIL_H__
#define __ZFD_AS_UTIL_H__


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

#include <Windows.h>
#include <direct.h>
#include<stdlib.h>
#include <iostream>

#define random(x) (rand()%x)

using namespace std;
using namespace cv;
using namespace cv::detail;


void least_square(Mat A, Mat b, Mat &x);

void my_drawMatches(Mat img1, Mat img2, vector<Point2f> src_pts, vector<Point2f> dst_pts, Mat &draw_img);

void my_drawMatches2(Mat img1, Mat img2, ImageFeatures feat1, ImageFeatures feat2, MatchesInfo matches_info, Mat &draw_img);

void getAllJpegs(string dirpath, std::vector <string> &filenames);

#endif