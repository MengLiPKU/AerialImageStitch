#ifndef __ZFD_AS_MATCH_H__
#define __ZFD_AS_MATCH_H__

#include "util.h"

using namespace std;
using namespace cv;
using namespace cv::detail;


void feature_match_bidirection(ImageFeatures feat1, ImageFeatures feat2, MatchesInfo &match_info);

void feature_match_bidirection_raw(ImageFeatures feat1, ImageFeatures feat2, MatchesInfo &matches_info, double match_conf_=0.3);

#endif