#ifndef __ZFD_AS_WARP_H__
#define __ZFD_AS_WARP_H__

#include "util.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

#define AS_SIMILAR 0
#define AS_AFFINE 1

int compute_transform(ImageFeatures feat1, ImageFeatures feat2, MatchesInfo matches_info, Mat &H, int trans_type, double thresh_error = 3);

int compute_perspective_transform(ImageFeatures feat1, ImageFeatures feat2, MatchesInfo matches_info, Mat &H, double thresh_error = 3);

Point2f warp_point(Point2f p, Mat H);

Rect warp_area(Mat H, Rect r);

Point my_warp_perspective(Mat src, Mat &dst, Mat &mask, Mat H);

#endif