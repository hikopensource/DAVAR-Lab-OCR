/*
##################################################################################################
// Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
// Filename       :    tp_points_generate.cpp
// Abstract       :    Generating fiducial points from predicted segmentation masks and regression masks

// Current Version:    1.0.0
// Date           :    2020-05-31
###################################################################################################
*/
#include <iostream>
#include <functional>
#include <utility>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cctype>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;

float DProduct(cv::Point a, cv::Point b, cv::Point c, cv::Point d);
bool InterSect(cv::Point a, cv::Point b, cv::Point c, cv::Point d);
void bfs_search(int                  h,
                int                  w,
                vector<vector<int> >&search_map,
                vector<cv::Point>   &points,
                vector<cv::Point>   &points1,
                vector<cv::Point>   &points2,
                int                 &out_back_num,
                int                 &out_bond_num,
                vector<vector<int> >&hasFind);
void bfs_search_bond(int                  h,
                     int                  w,
                     vector<vector<int> >&search_map,
                     vector<cv::Point>   &tmp_points,
                     int                  type,
                     vector<vector<int> >&hasFind);
bool completeContours(vector<cv::Point2f>&finalContour,
                      vector<cv::Point>  &points,
                      bool                horizon,
                      float               w_ratio,
                      float               h_ratio,
                      float              *geo_data,
                      int                 height,
                      int                 width,
                      int                 pointsNum_);

extern "C" void generate_result(float *score_pred_text,
                                float *score_pred_head,
                                float *score_pred_tail,
                                float *score_pred_bond,
                                float *geo_head_data,
                                float *geo_tail_data,
                                float *geo_bond_data,
                                int    height,
                                int    width,
                                int    pool_ratio,
                                float  scale_factor,
                                int    point_num,
                                float  filter_ratio,
                                float  thres_thext,
                                float  thres_head,
                                float  thres_bond,
                                int   *result,
                                int   *result_num);

float DProduct(cv::Point a, cv::Point b, cv::Point c, cv::Point d) {
    /*
        Calculate dot product for two vectors.

        Args:
            a: start point of vector 1.
            b: end point of vector 1.
            c: start point of vector 2.
            d: end point of vector 2.
        Returns:
            float: dot product of vector 1 and vector 2
    */
	return (b.x - a.x) * (d.y - c.y) - (d.x - c.x) * (b.y - a.y);
}
bool InterSect(cv::Point a, cv::Point b, cv::Point c, cv::Point d) {
    /*
        Judge if two line segments intersect.

        Args:
            a: start point of vector 1.
            b: end point of vector 1.
            c: start point of vector 2.
            d: end point of vector 2.
        Returns:
            bool: whether this two lines intersect.
    */
	float d1 = DProduct(a, b, a, c);
	float d2 = DProduct(a, b, a, d);
	float d3 = DProduct(c, d, c, a);
	float d4 = DProduct(c, d, c, b);
	if (d1 * d2 < 0 && d3 * d4 < 0) {
		return true;
	}else {
		return false;
	}
}

void bfs_search(int                  h,
                int                  w,
                vector<vector<int> >&search_map,
                vector<cv::Point>   &points,
                vector<cv::Point>   &points1,
                vector<cv::Point>   &points2,
                int                 &out_back_num,
                int                 &out_bond_num,
                vector<vector<int> >&hasFind)
{
    /*
        Using BFS to search the connected region, and store its head and tail region.

        Args:
            h: current searching y-coordinate
            w: current searching x-coordinate
            search_map：the map need to be searched
            points: used to store all center text region pixels
            points1：used to store all head region pixels
            points2：used to store all tail region pixels
            out_back_num: the number of background pixels around the center text region.
            out_bond_num：the number of boundary (including head and tail) around the center text region.
            hasFind：used to record whether a point has been searched
     */

    int height = search_map.size();
    int width  = search_map[0].size();

    // Search return if current point is out of range or it was searched before
    if ((w < 0) || (w > width - 1) || (h < 0) || (h > height - 1))
    {
        return;
    }
    if ((hasFind[h][w] == 1) || (search_map[h][w] == -1))
    {
        return;
    }

    if (search_map[h][w] == 0)
    {
        // If it is background pixel, accumulate out_back_num
        out_back_num++;
    }
    else if (search_map[h][w] == 1)
    {
        // If it is center text pixel, push it into points and continue to search in 4 directions.
        search_map[h][w] = -1;
        points.push_back(cv::Point(w, h));
        hasFind[h][w] = 1;
        bfs_search(h - 1, w, search_map, points, points1, points2, out_back_num, out_bond_num, hasFind);
        bfs_search(h + 1, w, search_map, points, points1, points2, out_back_num, out_bond_num, hasFind);
        bfs_search(h, w - 1, search_map, points, points1, points2, out_back_num, out_bond_num, hasFind);
        bfs_search(h, w + 1, search_map, points, points1, points2, out_back_num, out_bond_num, hasFind);
    }
    else if (search_map[h][w] == 4)
    {
        // If it is top&bottom boundary, accumulate out_bond_num
        out_bond_num++;
    }
    else if (search_map[h][w] == 2)
    {
        // If it is head region, accumulate out_bond_num, and start to search a head region using BFS
        out_bond_num++;
        hasFind[h][w] = 1;
        vector<cv::Point> tmp_points;
        tmp_points.push_back(cv::Point(w, h));
        bfs_search_bond(h - 1, w, search_map, tmp_points, 2, hasFind);
        bfs_search_bond(h + 1, w, search_map, tmp_points, 2, hasFind);
        bfs_search_bond(h, w - 1, search_map, tmp_points, 2, hasFind);
        bfs_search_bond(h, w + 1, search_map, tmp_points, 2, hasFind);

        // If current searched region is larger than previous one, replace it.
        if (tmp_points.size() > points1.size())
        {
            points1.clear();
            points1 = tmp_points;
        }
    }
    else if (search_map[h][w] == 3)
    {
        // If it is tail region, accumulate out_bond_num, and start to search a tail region using BFS
        out_bond_num++;
        hasFind[h][w] = 1;
        vector<cv::Point> tmp_points;
        tmp_points.push_back(cv::Point(w, h));
        bfs_search_bond(h - 1, w, search_map, tmp_points, 3, hasFind);
        bfs_search_bond(h + 1, w, search_map, tmp_points, 3, hasFind);
        bfs_search_bond(h, w - 1, search_map, tmp_points, 3, hasFind);
        bfs_search_bond(h, w + 1, search_map, tmp_points, 3, hasFind);

        // If current searched region is larger than previous one, replace it.
        if (tmp_points.size() > points2.size())
        {
            points2.clear();
            points2 = tmp_points;
        }
    }
}

void bfs_search_bond(int                  h,
                     int                  w,
                     vector<vector<int> >&search_map,
                     vector<cv::Point>   &tmp_points,
                     int                  type,
                     vector<vector<int> >&hasFind)
{
    /*
        BFS search for head (tail) region.

        Args:
            h: current searched point's y-coordinate
            w: current searched point's x-coordinate
            search_map: the map need to be searched
            tmp_points: used to store head (tail) region points
            type: 2 for head, 3 for tail
            hasFind：used to record whether a point has been searched
     */

    int height = search_map.size();
    int width  = search_map[0].size();

    // Search return if current point is out of range or it was searched before
    if ((w < 0) || (w > width - 1) || (h < 0) || (h > height - 1))
    {
        return;
    }
    if (hasFind[h][w] == 1)
    {
        return;
    }

    // Add the point into the set if it has the same type.
    if (search_map[h][w] == type)
    {
        hasFind[h][w] = 1;
        cv::Point p(w, h);
        tmp_points.push_back(p);

        // Continue BFS
        bfs_search_bond(h - 1, w, search_map, tmp_points, type, hasFind);
        bfs_search_bond(h + 1, w, search_map, tmp_points, type, hasFind);
        bfs_search_bond(h, w - 1, search_map, tmp_points, type, hasFind);
        bfs_search_bond(h, w + 1, search_map, tmp_points, type, hasFind);
    }
}


void completeContours(vector<cv::Point2f>&finalContour,
                      int                 start_index,
                      int                 end_index,
                      vector<cv::Point>  &points,
                      float               w_ratio,
                      float               h_ratio,
                      float              *geo_data,
                      int                 height,
                      int                 width)
{
    /*
        Calculate the other fiducial points in top &bottom boundary.

        Args：
            finalContour: used to store the final results
            start_index：current start point index
            end_index：current end point index
            points：set of center text
            w_ratio：the ratio used to calculate the original coordinate
            h_ratio：the ratio used to calculate the original coordinate
            geo_data：regression prediction map for center text region
            height: height of feature map
            width：width of feature map
     */
    // Return if start point and end point meets
    if (end_index - start_index <= 1)
    {
        return;
    }

    // Judge whether the connect line is horizontal or vertical
    float horizon = 1;

    if (abs(finalContour[end_index].x - finalContour[start_index].x) < 1e-5)
    {
        // If two points has the same x, prevent from division of 0
        horizon = 999;
    }
    else
    {
        // Calculate the slope
        horizon =
            abs((finalContour[end_index].y
                 - finalContour[start_index].y) / (finalContour[end_index].x - finalContour[start_index].x));
    }

    int mid_index = (end_index + start_index) / 2;
    if (horizon > 1)
    {
        // If it is a vertical text
        // Calculate the middle points
        float mid_x = finalContour[start_index].x
                      + (finalContour[end_index].x - finalContour[start_index].x) * (float)mid_index / (end_index + start_index);
        float mid_y = finalContour[start_index].y
                      + (finalContour[end_index].y - finalContour[start_index].y) * (float)mid_index / (end_index + start_index);
        vector<cv::Point> tmp;
        vector<cv::Point> top_part;
        vector<cv::Point> bottom_part;

        for (int m = 0; m < points.size(); m++)
        {
            // Push all pixels that has the close y-coordinate in center text region into a set (band region)
            if (abs(points[m].y * h_ratio - mid_y) <= 10)
            {
                tmp.push_back(points[m]);
            }

            // Split the center text pixels set into two parts
            if (points[m].y * w_ratio < mid_y)
            {
                top_part.push_back(points[m]);
            }
            else
            {
                bottom_part.push_back(points[m]);
            }
        }
        if (tmp.size() != 0)
        {
            float offset = 0;
            // Calculate the average x-coordinate for this fudicial point
            for (int m = 0; m < tmp.size(); m++)
            {
                int loc_h = tmp[m].y;
                int loc_w = tmp[m].x;
                if (finalContour[start_index].y > finalContour[end_index].y)
                {
                    offset += (loc_w * w_ratio - geo_data[loc_h * width + loc_w] * w_ratio * 8.);
                }
                else
                {
                    offset += (loc_w * w_ratio - geo_data[2 * height * width + loc_h * width + loc_w] * w_ratio * 8.);
                }
            }
            finalContour[mid_index] = cv::Point2f(offset / tmp.size(), mid_y);
        }
        else
        {
            finalContour[mid_index] = cv::Point2f(mid_x, mid_y);
        }

        // Continue the binary search
        if (finalContour[start_index].y < finalContour[end_index].y)
        {
            completeContours(finalContour, start_index, mid_index, top_part, w_ratio, h_ratio, geo_data, height, width);
            completeContours(finalContour, mid_index, end_index, bottom_part, w_ratio, h_ratio, geo_data, height, width);
        }
        else
        {
            completeContours(finalContour, start_index, mid_index, bottom_part, w_ratio, h_ratio, geo_data, height, width);
            completeContours(finalContour, mid_index, end_index, top_part, w_ratio, h_ratio, geo_data, height, width);
        }
    }
    else
    {
        // If it is a horizontal text
        // Calculate the middle points
        float mid_x = finalContour[start_index].x
                      + (finalContour[end_index].x - finalContour[start_index].x) * (float)mid_index / (end_index + start_index);
        float mid_y = finalContour[start_index].y
                      + (finalContour[end_index].y - finalContour[start_index].y) * (float)mid_index / (end_index + start_index);

        vector<cv::Point> tmp;
        vector<cv::Point> left_part;
        vector<cv::Point> right_part;
        for (int m = 0; m < points.size(); m++)
        {
            // Push all pixels that has the close x-coordinate in center text region into a set (band region)
            if (abs(points[m].x * w_ratio - mid_x) <= 10)
            {
                tmp.push_back(points[m]);
            }

            // Split the center text pixels set into two parts
            if (points[m].x * w_ratio < mid_x)
            {
                left_part.push_back(points[m]);
            }
            else
            {
                right_part.push_back(points[m]);
            }
        }
        if (tmp.size() != 0)
        {
            float offset = 0;
            for (int m = 0; m < tmp.size(); m++)
            {
                // Calculate the average x-coordinate for this fudicial point
                int loc_h = tmp[m].y;
                int loc_w = tmp[m].x;
                if (finalContour[start_index].x < finalContour[end_index].x)
                {
                    offset += (loc_h * h_ratio - geo_data[height * width + loc_h * width + loc_w] * h_ratio * 8.);
                }
                else
                {
                    offset += (loc_h * h_ratio - geo_data[3 * height * width + loc_h * width + loc_w] * h_ratio * 8.);
                }
            }
            finalContour[mid_index] = cv::Point2f(mid_x, offset / tmp.size());
        }
        else
        {
            finalContour[mid_index] = cv::Point2f(mid_x, mid_y);
        }
        // Continue the binary search
        if (finalContour[start_index].x < finalContour[end_index].x)
        {
            completeContours(finalContour, start_index, mid_index, left_part, w_ratio, h_ratio, geo_data, height, width);
            completeContours(finalContour, mid_index, end_index, right_part, w_ratio, h_ratio, geo_data, height, width);
        }
        else
        {
            completeContours(finalContour, start_index, mid_index, right_part, w_ratio, h_ratio, geo_data, height, width);
            completeContours(finalContour, mid_index, end_index, left_part, w_ratio, h_ratio, geo_data, height, width);
        }
    }
}

extern "C" void generate_result(float *score_pred_text,
                                float *score_pred_head,
                                float *score_pred_tail,
                                float *score_pred_bond,
                                float *geo_head_data,
                                float *geo_tail_data,
                                float *geo_bond_data,
                                int    height,
                                int    width,
                                int    pool_ratio,
                                float  scale_factor,
                                int    point_num,
                                float  filter_ratio,
                                float  thres_thext,
                                float  thres_head,
                                float  thres_bond,
                                int   *result,
                                int   *result_num)

/*
    The function that exposed to python. Using predicted feature maps to calculate the final fiducial points.

    Args:
        score_pred_text: the predicted feature map of center text region, length of [H*W]
        score_pred_head: the predicted feature map of head region, length of [H*W]
        score_pred_tail: the predicted feature map of tail region, length of [H*W]
        score_pred_bond: the predicted feature map of top&bottom boundary, length of [H*W]
        geo_head_data: the corner points offset prediction feature map for head region, length of [4*H*W]
        geo_tail_data: the corner points offset prediction feature map for tail region, length of [4*H*W]
        height: feature map height
        width: feature map width
        pool_ratio: the pool ratio feature map versus resized image
        scale_factor: the scale_factor of   resized image versus original image
        point_num: The final fiducial points count around the text
        filter_ratio: the parameter to filter out instances that with insufficient boundary pixels
        thres_text: the parameter to control center text pixels' recall
        thres_head: the parameter to control head and tail pixels' recall
        thres_bond: the parameter to contral top&bottom boundary pixels' recall
        result: used to store the final result
        result_num: used to store the number of final result
 */
{
    // Initialize the search map
    vector<vector<int> > map(height, vector<int>(width));
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            // Overlay the pixels orderly bond> tail > head> text
            if (score_pred_text[h * width + w] >= thres_thext)
            {
                map[h][w] = 1;
            }
            if (score_pred_head[h * width + w] >= thres_head)
            {
                map[h][w] = 2;
            }
            if (score_pred_tail[h * width + w] >= thres_head)
            {
                map[h][w] = 3;
            }
            if (score_pred_bond[h * width + w] >= thres_bond)
            {
                map[h][w] = 4;
            }
        }
    }
    // traverse the search map to search connected regions.
    float w_ratio = pool_ratio;
    float h_ratio = pool_ratio;
    *result_num = 0;

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            // only care about the center pixel region
            if (map[h][w] != 1)
            {
                continue;
            }

            int                  out_bond_num = 0, out_back_num = 0;
            vector<cv::Point>    points;   // Store center text pixels
            vector<cv::Point>    points_l; // Store head pixels
            vector<cv::Point>    points_r; // Store tail pixels
            vector<vector<int> > hasFind(height, vector<int>(width, 0));

            // Start BFS search
            bfs_search(h, w, map, points, points_l, points_r, out_back_num, out_bond_num, hasFind);

            // if the neighbor boundary pixels is insufficient, filter it out.
            if (((double)out_bond_num) / (out_bond_num + out_back_num) < filter_ratio)
            {
                continue;
            }

            // If there is no matched head or tail region, filter it out.
            if ((points_l.size() == 0) || (points_r.size() == 0) || (points.size() <= 2))
            {
                continue;
            }
            if (points_l == points_r)
            {
                continue;
            }

            // Used to store the final fiducial points
            vector<cv::Point2f> finalContour(point_num, cv::Point2f(-1, -1));

            // Calculate the top-left corner and bottom-left corner in head region.
            float top_avg_x    = 0;
            float top_avg_y    = 0;
            float bottom_avg_x = 0;
            float bottom_avg_y = 0;
            for (int i = 0; i < points_l.size(); i++)
            {
                int loc_w = points_l[i].x;
                int loc_h = points_l[i].y;
                top_avg_x    += (loc_w * w_ratio - geo_head_data[loc_h * width + loc_w] * w_ratio * 8.);
                top_avg_y    += (loc_h * h_ratio - geo_head_data[height * width + loc_h * width + loc_w] * h_ratio * 8.);
                bottom_avg_x += (loc_w * w_ratio - geo_head_data[2 * height * width + loc_h * width + loc_w] * w_ratio * 8.);
                bottom_avg_y += (loc_h * h_ratio - geo_head_data[3 * height * width + loc_h * width + loc_w] * h_ratio * 8.);
            }
            finalContour[0]             = cv::Point2f(top_avg_x / points_l.size(), top_avg_y / points_l.size());
            finalContour[point_num - 1] = cv::Point2f(bottom_avg_x / points_l.size(), bottom_avg_y / points_l.size());

            // Calculate the top-right corner and bottom-right corner in tail region.
            top_avg_x    = 0;
            top_avg_y    = 0;
            bottom_avg_x = 0;
            bottom_avg_y = 0;
            for (int i = 0; i < points_r.size(); i++)
            {
                int loc_w = points_r[i].x;
                int loc_h = points_r[i].y;
                top_avg_x    += (loc_w * w_ratio - geo_tail_data[loc_h * width + loc_w] * w_ratio * 8.);
                top_avg_y    += (loc_h * h_ratio - geo_tail_data[height * width + loc_h * width + loc_w] * w_ratio * 8.);
                bottom_avg_x += (loc_w * w_ratio - geo_tail_data[2 * height * width + loc_h * width + loc_w] * w_ratio * 8.);
                bottom_avg_y += (loc_h * h_ratio - geo_tail_data[3 * height * width + loc_h * width + loc_w] * w_ratio * 8.);
            }
            finalContour[point_num / 2 - 1] = cv::Point2f(top_avg_x / points_r.size(), top_avg_y / points_r.size());
            finalContour[point_num / 2]     = cv::Point2f(bottom_avg_x / points_r.size(), bottom_avg_y / points_r.size());

            cv::Rect rect = boundingRect(points);
			bool horizon = true;
			if(rect.height > rect.width){
				horizon = false;
			}

            // Calculate the other fiducial points
            completeContours(finalContour, 0, point_num / 2 - 1, points, w_ratio, h_ratio, geo_bond_data, height, width);
            completeContours(finalContour, point_num / 2, point_num - 1, points, w_ratio, h_ratio, geo_bond_data, height, width);


            // Scale points' coordinate into original img size
            for (int k = 0; k < finalContour.size(); k++)
            {
                result[point_num * 2 * (*result_num) + 2 * k]     = (int)(finalContour[k].x / scale_factor);
                result[point_num * 2 * (*result_num) + 2 * k + 1] = (int)(finalContour[k].y / scale_factor);
            }
            (*result_num) += 1;
        }
    }
}
