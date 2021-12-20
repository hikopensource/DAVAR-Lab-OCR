#include <iostream>
#include <functional>
#include <utility>
#include <vector>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cctype>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
void bfs_search(int h,
	            int w, 
	            vector<vector<int> >& seg_map,
	            vector<vector<int> >& cate_map,
	            vector<vector<int> >& visit,
				vector<Point>& points, 
				vector<int>& weight_map,
				set<int>& grids);
extern "C" void generate_result(float* score_pred_seg,
	                            float* score_pred_cate,
								int height, 
								int width,
								int grid_num,
								float thres_seg,
								float thres_cate,
								int* seg_result,
								int* cate_result,
								int* weight,
								int* result_num);
		  
void bfs_search(int h,
	            int w,
	            vector<vector<int> >& seg_map,
	            vector<vector<int> >& cate_map,
	            vector<vector<int> >& visit,
				vector<Point>& points,
				vector<int>& weight_map,
				set<int>& grids)
{
	int height = seg_map.size();
	int width = seg_map[0].size();
	int grid_num = cate_map.size();
	int grid_h = h * grid_num / height;
	int grid_w = w * grid_num / width;
	int grid_index = grid_h * grid_num + grid_w;
	if(w < 0 || w >= width || h < 0 || h >= height || visit[h][w] == 1 || seg_map[h][w] == 0) 
	{
		return;
	}
	if(cate_map[grid_h][grid_w] == 1)
	{
		grids.insert(grid_index);
		weight_map[grid_index] += 1;
	}
	visit[h][w] = 1;
	points.push_back(Point(w, h));
	bfs_search(h - 1, w, seg_map, cate_map, visit, points, weight_map, grids);
	bfs_search(h + 1, w, seg_map, cate_map, visit, points, weight_map, grids);
	bfs_search(h, w - 1, seg_map, cate_map, visit, points, weight_map, grids);
	bfs_search(h, w + 1, seg_map, cate_map, visit, points, weight_map, grids);
}
extern "C" void generate_result(float* score_pred_seg,
	                            float* score_pred_cate,
								int height, 
								int width,
								int grid_num,
								float thres_seg,
								float thres_cate,
								int* seg_result,
								int* cate_result,
								int* weight,
								int* result_num)
{
  
	// Initialization
	vector<vector<int> > seg_map(height,vector<int>(width));
	vector<vector<int> > cate_map(grid_num, vector<int>(grid_num));
	vector<vector<int> > visit(height, vector<int>(width));

	*result_num = 0;
	set<int>::iterator it;
	
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(score_pred_seg[h * width + w] >= thres_seg)
			{
				seg_map[h][w] = 1;
			}
		}
	}

	for(int h = 0; h < grid_num; h++)
	{
		for(int w = 0; w < grid_num; w++)
		{
			if(score_pred_cate[h * grid_num + w] >= thres_cate)
			{
				cate_map[h][w] = 1;
			}
		}
	}

	// Search connected blocks and their corresponding grids
	int p = 0;
	int cnt = 0;
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(visit[h][w] != 0)
			{
				continue;
			}
			// Used to store text pixels
			vector<Point> points; 

			// Used to store the corresponding weights
			vector<int> weight_map(grid_num * grid_num); 

			// Used to store text pixels
			set<int> grids; 
			bfs_search(h, w, seg_map, cate_map, visit, points, weight_map, grids);
			if(points.size() <= 8)
			{
				continue;
			}

			if(grids.size() == 0)
			{
				continue;
			}

			RotatedRect r = minAreaRect(points); // Generate the smallest enclosing rectangle
			Point2f vertex[4];
			r.points(vertex);
			for(int i = 0; i < 4; i++)
			{
				seg_result[p++] = vertex[i].x;
				seg_result[p++] = vertex[i].y;
			}
			for(it = grids.begin(); it != grids.end(); it++)
			{
				// If two instances occupy the same grid, the grid with the larger iou is selected
				if(weight_map[*it] > weight[*it])
				{
					weight[*it] = weight_map[*it];
					cate_result[*it] = *result_num + 1;
				}
			}
			(*result_num) += 1;
		}
	}
}
