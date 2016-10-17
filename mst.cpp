
#include "mst.h"
#include <stdlib.h>

using namespace std;

MST::MST( vector<vector<MSTEdge>> graph_edges )
{
	int n_nodes = graph_edges.size();
	this->graph_edges_ = graph_edges;
	
	this->nodes.resize(n_nodes);
	this->edges.resize(n_nodes);
	for(int i = 0; i < n_nodes; i++)
	{
		this->nodes[i].index = i;
		this->nodes[i].parent = &(this->nodes[i]);
		this->edges[i].resize(n_nodes);
		for(int j = 0; j < n_nodes; j++)
			this->edges[i][j] = false;
	}
	cout << n_nodes << endl;
}

static int mst_edge_cmp(const void *a , const void *b)
{
	MSTEdge *e1 = (MSTEdge *)a;
	MSTEdge *e2 = (MSTEdge *)b;
	return (e1->weight > e2->weight) ? 1 : -1;
}

static MSTNode *find_root(MSTNode *node)
{
	MSTNode *ptr = node;
	while(ptr->parent != ptr)
		ptr = ptr->parent;
	return ptr;
}

void MST::build()
{
	int n_nodes = graph_edges_.size();
	vector<MSTEdge> graph_edges;
	graph_edges.clear();
	for(int i = 0; i < n_nodes; i++)
		for(int j = 0; j < n_nodes; j++)
			if(graph_edges_[i][j].exist)
				graph_edges.push_back(graph_edges_[i][j]);
	
	int edge_num = graph_edges.size();
	qsort(graph_edges.data(), edge_num, sizeof(MSTEdge), mst_edge_cmp);
	cout << edge_num << " edges" << endl;

	for(int ei = 0; ei < edge_num; ei++)
	{
		int ni = graph_edges[ei].i;
		int nj = graph_edges[ei].j;
		MSTNode *root_i = find_root(&(nodes[ni]));
		MSTNode *root_j = find_root(&(nodes[nj]));
		
		if(root_i != root_j)
		{
			edges[ni][nj] = true;
			edges[nj][ni] = true;
			cout << ni << "->" << root_i->index << ", " << nj << "->" << root_j->index << ". weight=" << graph_edges[ei].weight << endl;
			root_j->parent = root_i;
		}
	}
	cout << "build over" << endl;
}

int MST::get_root()
{
	int n_nodes = graph_edges_.size();
	vector<int> degrees(n_nodes);
	vector<double> weights(n_nodes);
	for(int i = 0; i < n_nodes; i++)
	{
		degrees[i] = 0;
		weights[i] = 0;
	}
	for(int i = 0; i < n_nodes; i++)
		for(int j = 0; j < n_nodes; j++)
			if(edges[i][j])
			{
				degrees[i]++;
				weights[i] += graph_edges_[i][j].weight;
			}
	int root_idx = 0, max_degree = degrees[0];
	double max_weight = weights[0];

	for(int i = 0; i < n_nodes; i++)
		cout << degrees[i] << ", ";
	cout << endl;

	for(int i = 1; i < n_nodes; i++)
	{
		if(degrees[i] > max_degree)
		{
			max_degree = degrees[i];
			root_idx = i;
		}
	}
	return root_idx;
}

