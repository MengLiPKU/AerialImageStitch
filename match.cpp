
#include "match.h"
#include <set>

using namespace std;
using namespace cv;
using namespace cv::detail;

static void get_match_dst_feat_index(ImageFeatures feat, MatchesInfo match_info, vector<int> &indices)
{
	int feat_num = feat.keypoints.size();
	indices.resize(feat_num);
	for(int i = 0; i < feat_num; i++)
		indices[i] = -1;
	//cout << feat_num << endl;

	// bug of opencv
	if (match_info.inliers_mask.size() != match_info.matches.size())
		return;

	for(int k = 0; k < match_info.matches.size(); k ++)
	{
		//cout << "a" << match_info.inliers_mask.size() << "b" << match_info.matches.size();
		const DMatch& m = match_info.matches[k];
		if (match_info.inliers_mask[k])
		{
			//cout << "(" << m.queryIdx << ", " << m.trainIdx << ") ";
			indices[m.queryIdx] = m.trainIdx;
		}
	}
	//cout << endl;
}

static void check_matchinfo(MatchesInfo &match_info)
{
	int n = match_info.matches.size();
	if(match_info.inliers_mask.size() < n)
	{
		match_info.inliers_mask.resize(n, false);
	}
}

void feature_match_bidirection(ImageFeatures feat1, ImageFeatures feat2, MatchesInfo &match_info)
{
	BestOf2NearestMatcher matcher(false, 0.3f);
	MatchesInfo match_info21;
	matcher(feat1, feat2, match_info);
	check_matchinfo(match_info);
	matcher(feat2, feat1, match_info21);
	check_matchinfo(match_info21);
	//cout << "a:" << match_info.matches.size() << ", " << match_info21.matches.size() << endl;
	vector<int> indices12, indices21;
	get_match_dst_feat_index(feat1, match_info, indices12);
	get_match_dst_feat_index(feat2, match_info21, indices21);
	//cout << "b" << endl;
	int inliner_num = 0;
	for(int k = 0; k < match_info.matches.size(); k ++)
	{
		const DMatch& m = match_info.matches[k];
		if (match_info.inliers_mask[k])
		{
			if(indices21[m.trainIdx] == m.queryIdx)
				inliner_num++;
			else
				match_info.inliers_mask[k] = 0;
		}
	}
	//cout << "c" << endl;
	match_info.num_inliers = inliner_num;
	matcher.collectGarbage();
	match_info21.matches.clear();
	match_info21.inliers_mask.clear();
}

void feature_match_bidirection_raw(ImageFeatures feat1, ImageFeatures feat2, MatchesInfo &matches_info, double match_conf_)
{
	CV_Assert(feat1.descriptors.type() == feat2.descriptors.type());
	CV_Assert(feat2.descriptors.depth() == CV_8U || feat2.descriptors.depth() == CV_32F);

	matches_info.matches.clear();
	Ptr<flann::IndexParams> indexParams = new flann::KDTreeIndexParams();
	Ptr<flann::SearchParams> searchParams = new flann::SearchParams();

	if (feat2.descriptors.depth() == CV_8U)
	{
		indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
		searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
	}

	FlannBasedMatcher matcher(indexParams, searchParams);
	vector< vector<DMatch> > pair_matches;
	set<pair<int,int> > matches;
	
	// Find 1->2 matches
	matcher.knnMatch(feat1.descriptors, feat2.descriptors, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];
		if (m0.distance < (1.f - match_conf_) * m1.distance)
			matches.insert(make_pair(m0.queryIdx, m0.trainIdx));
	}
	
	// Find 2->1 matches
	pair_matches.clear();
	matcher.knnMatch(feat2.descriptors, feat1.descriptors, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];
		if (m0.distance < (1.f - match_conf_) * m1.distance)
			if (matches.find(make_pair(m0.trainIdx, m0.queryIdx)) != matches.end())
				matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
	}
	int num_inliers = matches_info.matches.size();
	
	matches_info.inliers_mask.resize(num_inliers);
	for(int i = 0; i < num_inliers; i++)
		matches_info.inliers_mask[i] = 1;
	matches_info.num_inliers = num_inliers;

	matcher.clear();
	pair_matches.clear();
	matches.clear();
}