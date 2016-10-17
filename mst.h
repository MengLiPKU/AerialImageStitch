#ifndef __ZFD_AS_MST_H__
#define __ZFD_AS_MST_H__

#include "util.h"

using namespace std;

class MSTNode
{
public:
	int index;
	MSTNode *parent;
};

#define MST_MAX_WEIGHT 100000000

class MSTEdge
{
public:
	int i;
	int j;
	bool exist;
	double weight;
	MSTEdge()
	{
		exist = false;
		weight = MST_MAX_WEIGHT;
	}
	void set_value(int i_, int j_, bool exist_, double weight_)
	{
		i = i_;
		j = j_;
		exist = exist_;
		weight = weight_;
	}
};

class MST
{
public:
	vector<MSTNode> nodes;
	vector<vector<bool>> edges;

	MST(vector<vector<MSTEdge>> graph_edges);
	void build();
	int get_root();

private:
	vector<vector<MSTEdge>> graph_edges_;
};

#endif