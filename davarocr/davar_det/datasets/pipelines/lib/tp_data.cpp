/*
##################################################################################################
// Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
// Filename       :    tp_data.cpp
// Abstract       :    GT_mask generating in Text Perceptron

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
#include <math.h>
#include <cctype>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <opencv2/opencv.hpp>

using namespace std;

float DProduct(cv::Point a,
               cv::Point b,
               cv::Point c,
               cv::Point d);

vector<vector<cv::Point> > Border_Poly(vector<cv::Point> poly,
                                       float             ratio1,
                                       float             ratios);
float PointDistance(cv::Point a,
                    cv::Point b);
void balance_BF(int  *conf_data,
                float fg_ratio,
                int  *mask_data,
                int   out_height_,
                int   out_width_);

vector<cv::Point> get_poly(vector<cv::Point>&points,
                           vector<int>      &quad,
                           int               index,
                           float             min_size);
float getAngle(cv::Point pt1,
               cv::Point pt2,
               cv::Point c);

vector<int> getQuadIndex(vector<cv::Point> points);
void quick_sort(vector<float>&angles,
                vector<int>  &index,
                int           low,
                int           high);

extern "C" void parse_tp_data(int    height,
                              int    width,
                              int   *gt_boxes,
                              int    gt_boxes_size,
                              int   *gt_boxes_length,
                              int   *gt_boxes_ignore,
                              int    gt_boxes_ignore_size,
                              int   *gt_boxes_ignore_length,
                              int    pool_ratio,
                              float  head_ratio,
                              float  bond_ratio,
                              float  ignore_ratio,
                              int   *gt_score_map,
                              int   *gt_mask,
                              float *gt_geo_head,
                              float *gt_geo_head_weight,
                              float *gt_geo_tail,
                              float *gt_geo_tail_weight,
                              float *gt_geo_bond,
                              float *gt_geo_bond_weight);

float DProduct(cv::Point a,
               cv::Point b,
               cv::Point c,
               cv::Point d)
{
    /*
        Calculate the cross product of two vectors
     */
    return (b.x - a.x) * (d.y - c.y) - (d.x - c.x) * (b.y - a.y);
}

float PointDistance(cv::Point a,
                    cv::Point b)
{
    /*
        Calculate the point distance
     */
    return sqrt((double)(pow((b.x - a.x), 2.0) + pow((b.y - a.y), 2)));
}

float getAngle(cv::Point pt1,
               cv::Point pt2,
               cv::Point c)
{
    /*
        Calculate the angle of ∠pt1,C,pt2
     */
    float theta = atan2(pt1.x - c.x, pt1.y - c.y) - atan2(pt2.x - c.x, pt2.y - c.y);

    if (theta > CV_PI)
    {
        theta -= 2 * CV_PI;
    }
    if (theta < -CV_PI)
    {
        theta += 2 * CV_PI;
    }
    theta = theta * 180.0 / CV_PI;
    if (theta < 0)
    {
        theta = -theta;
    }
    if (DProduct(pt1, c, c, pt2) < 0)
    {
        theta = 360 - theta;
    }
    return theta;
}

vector<cv::Point> get_poly(vector<cv::Point>&points,
                           vector<int>      &quad,
                           int               index,
                           float             min_size,
                           float             shrink_ratio_)
{
    /*
        According to the identified 4 corner points, generate the head and tail regionss

        Args
            points: points of the polygon boxes. [ (x1, y1), ..., (xn, yn)]
            quad: indexes to identify the 4 corner points, e.g., [0, 6, 7, 13]
            index: set 1 to generate head region; 2 to generate tail region.
            min_size: minimum length of boundaries
            shrink_ratio: parameter to control the width of the generated region.
        Returns:
            ret: a quadrangle region to represent head or tail region, [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
     */
    vector<cv::Point> ret(4);

    if (index == 1)
    {
        // Head region
        ret[0] = points[quad[0]];
        ret[3] = points[quad[3]];

        float theta = atan2(points[(quad[0] + 1) % points.size()].y - points[quad[0]].y,
                            points[(quad[0] + 1) % points.size()].x - points[quad[0]].x);
        ret[1].x = ret[0].x + shrink_ratio_ * min_size * cos(theta);
        ret[1].y = ret[0].y + shrink_ratio_ * min_size * sin(theta);
        theta    = atan2(points[(quad[3] - 1) % points.size()].y - points[quad[3]].y,
                         points[(quad[3] - 1) % points.size()].x - points[quad[3]].x);
        ret[2].x = ret[3].x + shrink_ratio_ * min_size * cos(theta);
        ret[2].y = ret[3].y + shrink_ratio_ * min_size * sin(theta);
    }
    else if (index == 2)
    {
        // Tail region
        ret[1] = points[quad[1]];
        ret[2] = points[quad[2]];

        float theta =
            atan2(points[quad[1]].y
                  - points[(quad[1] - 1) % points.size()].y, points[quad[1]].x - points[(quad[1] - 1) % points.size()].x);
        ret[0].x = ret[1].x - shrink_ratio_ * min_size * cos(theta);
        ret[0].y = ret[1].y - shrink_ratio_ * min_size * sin(theta);
        theta    =
            atan2(points[quad[2]].y
                  - points[(quad[2] + 1) % points.size()].y, points[quad[2]].x - points[(quad[2] + 1) % points.size()].x);
        ret[3].x = ret[2].x - shrink_ratio_ * min_size * cos(theta);
        ret[3].y = ret[2].y - shrink_ratio_ * min_size * sin(theta);
    }
    return ret;
}

void balance_BF(int  *conf_data,
                float fg_ratio,
                int  *mask_data,
                int   out_height_,
                int   out_width_)
{
    /*
        Balance foreground and background pixels. Set some of the background pixels' weight to 0 according to the fg_ratio

        Args:
            conf_data: score_map of the segmentation
            fg_ratio: the ratio of foreground pixels / background pixels
            mask_data: the mask (weight) of score_map
            out_height_: height of score_map
            out_width_: width of score_map
     */

    vector<int> fg_inds;  // Store foreground pixels index
    vector<int> bg_inds;  // Store background pixels index

    for (int h = 0; h < out_height_; h++)
    {
        for (int w = 0; w < out_width_; w++)
        {
            int index = h * out_width_ + w;
            if (conf_data[index] == 0)
            {
                bg_inds.push_back(index);
            }
            else if (conf_data[index] > 0)
            {
                fg_inds.push_back(index);
            }
        }
    }

    int num_fg_all = fg_inds.size();
    int num_bg_all = bg_inds.size();
    if (num_bg_all * fg_ratio > num_fg_all)
    {
        // Randomly select background pixels
        random_shuffle(bg_inds.begin(), bg_inds.end());
        for (int i = 0; i < num_bg_all * fg_ratio - num_fg_all; i++)
        {
            if (mask_data != NULL)
            {
                mask_data[bg_inds[i]] = 0;
            }
        }
    }
}

vector<int> getQuadIndex(vector<cv::Point> points)
{
    /*
        Estimate the corner points indexes. Make the top-left point as the index 0.
        For vertical instances, make the right-left point as the index 0.
        e.g., for quadrangle, the order is top-lef, top-right, bottom-right, bottom-left, respectively.

        Args:
            points: points of the polygon boxes. [ (x1, y1), ..., (xn, yn)]
        Return:
            quad: re-ordered corner points indexes,
     */

    // If the original annotation is 4 points, they were labeled from top-left corner clockwise.
    if (points.size() == 4)
    {
        vector<int> tmp(4);
        tmp[0] = 0;
        tmp[1] = 1;
        tmp[2] = 2;
        tmp[3] = 3;
        // If the text is a vertical text, change the first corner as top-right corner.
        if (PointDistance(points[0], points[3]) > 2 * PointDistance(points[0], points[1]))
        {
            tmp[0] = 1;
            tmp[1] = 2;
            tmp[2] = 3;
            tmp[3] = 0;
        }
        return tmp;
    }

    vector<float> angles;
    vector<int>   index;

    // The neighbor boundaries of two corner points are parallel
    for (int i = 0; i < points.size(); i++)
    {
        float angle1 = getAngle(points[(i - 1 + points.size()) % points.size()], points[(i + 1) % points.size()], points[i]);
        float angle2 =
            getAngle(points[(i - 2 + points.size()) % points.size()], points[i],
                     points[(i - 1 + points.size()) % points.size()]);
        // float angle = getAngle(points[i-1], points[(i+1)%points.size()], points[i]);
        angles.push_back(abs(angle1 + angle2 - 180.0));
        index.push_back(i);
    }

    // Sort all the angles to find the least angles that except the top-left corner.
    quick_sort(angles, index, 0, angles.size() - 1);
    vector<int> ret(4, 0);

    // t is the index of the bottom-right corner
    int t = 1;
    while (abs(index[0] - index[t]) == 1 or abs(index[0] - index[t]) == points.size() - 1)
    {
        t++;
        if (t == index.size())
        {
            // Annotation error
            return ret;
        }
    }

    if (index[0] < index[t])
    {
        // index[0] is the top-left corner, index[t] is the bottom-right corner
        ret[0] = index[0];
        ret[1] = (index[t] - 1 + points.size()) % points.size();
        ret[2] = index[t];
        ret[3] = (index[0] - 1 + points.size()) % points.size();
    }
    else
    {
        // index[t] is the top-left corner, index[0] is the bottom-right corner
        ret[0] = index[t];
        ret[1] = (index[0] - 1 + points.size()) % points.size();
        ret[2] = index[0];
        ret[3] = (index[t] - 1 + points.size()) % points.size();
    }

    return ret;
}

void quick_sort(vector<float>&angles,
                vector<int>  &index,
                int           low,
                int           high)
{
    /*
        Quick Sort angles along with its index depend the angles.
     */
    if (low >= high)
    {
        return;
    }
    int   first     = low;
    int   last      = high;
    float key       = angles[first];
    int   index_key = index[first];

    while (first < last)
    {
        while (first < last && angles[last] >= key)
        {
            --last;
        }
        angles[first] = angles[last];
        index[first]  = index[last];
        while (first < last && angles[first] <= key)
        {
            ++first;
        }
        angles[last] = angles[first];
        index[last]  = index[first];
    }
    angles[first] = key;
    index[first]  = index_key;

    quick_sort(angles, index, low, first - 1);
    quick_sort(angles, index, first + 1, high);
}

extern "C" void parse_tp_data(int    height,
                              int    width,
                              int   *gt_boxes,
                              int    gt_boxes_size,
                              int   *gt_boxes_length,
                              int   *gt_boxes_ignore,
                              int    gt_boxes_ignore_size,
                              int   *gt_boxes_ignore_length,
                              int    pool_ratio,
                              float  head_ratio,
                              float  bond_ratio,
                              float  ignore_ratio,
                              int   *gt_score_map,
                              int   *gt_mask,
                              float *gt_geo_head,
                              float *gt_geo_head_weight,
                              float *gt_geo_tail,
                              float *gt_geo_tail_weight,
                              float *gt_geo_bond,
                              float *gt_geo_bond_weight)
{
    /*
        The function that exposed to python.
        According to the annotations of polygon points, generate the segmentation score_map and regression geo-map

        Args:
            height: height of the original image, int.
            width: width of the original image, int
            gt_boxes: the ground_thruth of polygon points [[x1,y1,x2,y2,...],...]
            gt_boxes_size: the length of gt_boxes, int
            gt_boxes_length: the length of each box in gt_boxes, [8, 14, 8, ..., 12]
            gt_boxes_ignore: the ground_thruth of polygon points [[x1,y1,x2,y2,...],...]
            gt_boxes_ignore_size: the length of gt_boxes, int
            gt_boxes_ignore_length: the length of each box in gt_boxes, [8, 14, 8, ..., 12]
            pool_ratio: the downsample ratio of feature map versus original image.
            head_ratio: parameters to control the width of head/tail regions
            bond_ratio: parameter to control the width of top/bottom regions
            ignore_ratio: parameter to control the ratio of foreground pixels and background pixels.

        Return:
            gt_score_map: the segmentation score map [H, W]
            gt_mask:  the mask(weight) of segmentation score map [H, W]
            gt_geo_head: the regression score map of head region [4, H, W]
            gt_geo_head_weight: the mask(weight) of regression map of head region, [4, H, W]
            gt_geo_tail: the regression score map of tail region [4, H, W]
            gt_geo_tail_weight: the mask(weight) of regression map of tail region, [4, H, W]
            gt_geo_bond: the regression score map of top&bottom boundary region [4, H, W]
            gt_geo_bond_weight: the mask(weight) of regression map of top&bottom boundary region, [4, H, W]
     */

    // Initialize an empty mat
    vector<vector<cv::Point> > draw_tmp;
    cv::Mat                    poly_mask_origin_text = cv::Mat::zeros(height, width, CV_32FC1);
    int                        out_height_           = height / pool_ratio; // feature map height
    int                        out_width_            = width / pool_ratio;  // feature map width

    // Initialize the return value
    for (int i = 0; i < out_height_ * out_width_; i++)
    {
        gt_score_map[i] = 0;
        gt_mask[i]      = 1;
    }
    for (int i = 0; i < 4 * out_height_ * out_width_; i++)
    {
        gt_geo_head[i]        = 0;
        gt_geo_head_weight[i] = 0;
        gt_geo_tail[i]        = 0;
        gt_geo_tail_weight[i] = 0;
        gt_geo_bond[i]        = 0;
        gt_geo_bond_weight[i] = 0;
    }

    // Draw all of the ignore polygon regions with 64, and mask with 0
    draw_tmp.clear();
    for (int i = 0; i < gt_boxes_ignore_size; i++)
    {
        vector<cv::Point> poly;
        for (int j = 0; j < gt_boxes_ignore_length[i]; j += 2)
        {
            poly.push_back(cv::Point(gt_boxes_ignore[i * 48 + j], gt_boxes_ignore[i * 48 + j + 1]));
        }
        draw_tmp.push_back(poly);
    }
    cv::fillPoly(poly_mask_origin_text, draw_tmp, 64);
    for (int h = 0; h < out_height_; h++)
    {
        for (int w = 0; w < out_width_; w++)
        {
            if (poly_mask_origin_text.at<float>(int(pool_ratio * h), int(pool_ratio * w)) == 64)
            {
                gt_mask[h * out_width_ + w] = 0;
            }
        }
    }

    // Draw all of the cared polygon regions with 1.
    for (int i = 0; i < gt_boxes_size; i++)
    {
        draw_tmp.clear();
        cv::Mat           poly_mask_origin = cv::Mat::zeros(height, width, CV_32FC1);
        vector<cv::Point> points;
        for (int j = 0; j < gt_boxes_length[i]; j += 2)
        {
            points.push_back(cv::Point(gt_boxes[i * 48 + j], gt_boxes[i * 48 + j + 1]));
        }

        draw_tmp.push_back(points);
        cv::fillPoly(poly_mask_origin, draw_tmp, 1);

        // Estimate the four corner points
        // ------------------------------------Total-Text-------------------------------------------
		vector<int> quad = getQuadIndex(points);
		if(quad[0] == quad[1] || quad[0] == quad[2] || quad[3] == quad[1] || quad[3] == quad[2]){
			for (int h = 0; h < out_height_; h++) {
				for (int w = 0; w < out_width_; w++) {
					if(poly_mask_origin_text.at<float>(int(pool_ratio*h), int(pool_ratio*w)) == 1){
						gt_mask[h*out_width_ + w] = 0;
					}
				}
			}
			continue;
		}
        // --------------------------------------CTW--------------------------------------------------
     //   vector<int> quad(4, 0);
     //   quad[0] = 0;
     //   quad[1] = 6;
     //   quad[2] = 7;
     //   quad[3] = 13;
        // ---------------------------------------------------------------------------------------------

        // Get the shortest boundary length
        float min_size = 9999;
        for (int i = 0; i < points.size(); i++)
        {
            min_size = std::min(min_size, (float)PointDistance(points[i], points[(i + 1) % points.size()]));
        }
        // Calculate head and tail regions.
        vector<cv::Point> head_region;
        vector<cv::Point> tail_region;
        head_region = get_poly(points, quad, 1, std::max((double)min_size, (double)12.0), head_ratio);
        tail_region = get_poly(points, quad, 2, std::max((double)min_size, (double)12.0), head_ratio);

        // Draw head region with 2
        draw_tmp.clear();
        draw_tmp.push_back(head_region);
        cv::fillPoly(poly_mask_origin, draw_tmp, 2);

        // Draw tail region with 3
        draw_tmp.clear();
        draw_tmp.push_back(tail_region);
        cv::fillPoly(poly_mask_origin, draw_tmp, 3);

        // Draw top boundary region with 4
        for (int j = quad[0]; j < ((quad[1] > quad[0]) ? quad[1] : (quad[1] + points.size())); j++)
        {
            line(poly_mask_origin, points[j % points.size()], points[(j + 1) % points.size()], 4,
                 std::min(std::max((float)min_size * bond_ratio, (float)3.0), (float)12.0));
        }
        // Draw bottom boundary region with 4
        for (int j = quad[2]; j < ((quad[3] > quad[2]) ? quad[3] : (quad[3] + points.size())); j++)
        {
            line(poly_mask_origin, points[j % points.size()], points[(j + 1) % points.size()], 4,
                 std::min(std::max((float)min_size * bond_ratio, (float)3.0), (float)12.0));
        }

        // Set the weight param to prevent the L1 loss become too big.
        float weight_param = (pool_ratio * pool_ratio) / (256 * 64.0);
        for (int h = 0; h < out_height_; h++)
        {
            for (int w = 0; w < out_width_; w++)
            {
                if (poly_mask_origin.at<float>(int(pool_ratio * h), int(pool_ratio * w)) == 1)
                {
                    // Set all pixels with 1 as center text region
                    gt_score_map[h * out_width_ + w] = 1;

                    // Generate the center text to boundaries offset.
                    for (int k = 0; k < points.size(); k++)
                    {
						// The cross point is between index k and k+1
                        if (((w * pool_ratio > points[k].x)
                             && (w * pool_ratio <= points[(k + 1) % points.size()].x))
                            || ((w * pool_ratio < points[k].x) && (w * pool_ratio >= points[(k + 1) % points.size()].x)))
                        {
                            float offset_y = points[k].y
                                             + (w * pool_ratio
                                                - points[k].x)
                                             * (points[(k
                                                        + 1)
                                                       % points.size()].y
                                                - points[k].y) / (points[(k + 1) % points.size()].x - points[k].x);
                            if (offset_y <= h * pool_ratio)
                            {
                                // Calculate the offsets of point [h,w] to its nearest top boundary (y-direction).
                                if ((gt_geo_bond[1 * out_height_ * out_width_ + h * out_width_ + w] == 0)
                                    || (abs((h * pool_ratio - offset_y) / (pool_ratio * 8.)) <
                                        abs(gt_geo_bond[1 * out_height_ * out_width_ + h * out_width_ + w])))
                                {
                                    gt_geo_bond[1 * out_height_ * out_width_ + h * out_width_
                                                + w] = (h * pool_ratio - offset_y) / (pool_ratio * 8.);
                                    gt_geo_bond_weight[1 * out_height_ * out_width_ + h * out_width_ + w] = weight_param;
                                }
                            }
                            else
                            {
                                // Calculate the offsets of point [h,w] to its nearest bottom boundary (y-direction)
                                if ((gt_geo_bond[3 * out_height_ * out_width_ + h * out_width_ + w] == 0)
                                    || (abs((h * pool_ratio - offset_y) / (pool_ratio * 8.)) <
                                        abs(gt_geo_bond[3 * out_height_ * out_width_ + h * out_width_ + w])))
                                {
                                    gt_geo_bond[3 * out_height_ * out_width_ + h * out_width_
                                                + w] = (h * pool_ratio - offset_y) / (pool_ratio * 8.);
                                    gt_geo_bond_weight[3 * out_height_ * out_width_ + h * out_width_ + w] = weight_param;
                                }
                            }
                        }
                        if (((h * pool_ratio > points[k].y)
                             && (h * pool_ratio <= points[(k + 1) % points.size()].y))
                            || ((h * pool_ratio < points[k].y) && (h * pool_ratio >= points[(k + 1) % points.size()].y)))
                        {
                            float offset_x = points[k].x
                                             + (h * pool_ratio
                                                - points[k].y)
                                             * (points[(k
                                                        + 1)
                                                       % points.size()].x
                                                - points[k].x) / (points[(k + 1) % points.size()].y - points[k].y);
                            if (offset_x <= w * pool_ratio)
                            {
                                // Calculate the offsets of point [h,w] to its nearest left boundary. (x-direction)
                                if ((gt_geo_bond[0 * out_height_ * out_width_ + h * out_width_ + w] == 0)
                                    || (abs((w * pool_ratio - offset_x) / (pool_ratio * 8.)) <
                                        abs(gt_geo_bond[0 * out_height_ * out_width_ + h * out_width_ + w])))
                                {
                                    gt_geo_bond[0 * out_height_ * out_width_ + h * out_width_
                                                + w] = (w * pool_ratio - offset_x) / (pool_ratio * 8.);
                                    gt_geo_bond_weight[0 * out_height_ * out_width_ + h * out_width_ + w] = weight_param;
                                }
                            }
                            else
                            {
                                //  Calculate the offsets of point [h,w] to its nearest right boundary. (x-direction)
                                if ((gt_geo_bond[2 * out_height_ * out_width_ + h * out_width_ + w] == 0)
                                    || (abs((w * pool_ratio - offset_x) / (pool_ratio * 8.)) <
                                        abs(gt_geo_bond[2 * out_height_ * out_width_ + h * out_width_ + w])))
                                {
                                    gt_geo_bond[2 * out_height_ * out_width_ + h * out_width_
                                                + w] = (w * pool_ratio - offset_x) / (pool_ratio * 8.);
                                    gt_geo_bond_weight[2 * out_height_ * out_width_ + h * out_width_ + w] = weight_param;
                                }
                            }
                        }
                    }
                }

                else if (poly_mask_origin.at<float>(int(pool_ratio * h), int(pool_ratio * w)) == 2)
                {
                    // Set all pixels with 2 as head region
                    gt_score_map[h * out_width_ + w] = 2;

                    // Calculate the offset of pixels in head region to the top-left corner and bottom-left corner
                    gt_geo_head[h * out_width_ + w] = (pool_ratio * w - points[quad[0]].x) / (pool_ratio * 8.);
                    gt_geo_head[out_height_ * out_width_ + h * out_width_
                                + w] = (h * pool_ratio - points[quad[0]].y) / (pool_ratio * 8.);
                    gt_geo_head[2 * out_height_ * out_width_ + h * out_width_
                                + w] = ((pool_ratio * w - points[quad[3]].x) / (pool_ratio * 8.));
                    gt_geo_head[3 * out_height_ * out_width_ + h * out_width_
                                + w] = ((h * pool_ratio - points[quad[3]].y) / (pool_ratio * 8.));
                    gt_geo_head_weight[h * out_width_ + w] = weight_param;
                    gt_geo_head_weight[out_height_ * out_width_ + h * out_width_ + w]     = weight_param;
                    gt_geo_head_weight[2 * out_height_ * out_width_ + h * out_width_ + w] = weight_param;
                    gt_geo_head_weight[3 * out_height_ * out_width_ + h * out_width_ + w] = weight_param;
                }
                else if (poly_mask_origin.at<float>(int(pool_ratio * h), int(pool_ratio * w)) == 3)
                {
                    // Set all pixels with 3 as tail region
                    gt_score_map[h * out_width_ + w] = 3;

                    // Calculate the offset of pixels in tail region to the top-right corner and bottom-right corner
                    gt_geo_tail[h * out_width_ + w] = (pool_ratio * w - points[quad[1]].x) / (pool_ratio * 8.);
                    gt_geo_tail[out_height_ * out_width_ + h * out_width_
                                + w] = (h * pool_ratio - points[quad[1]].y) / (pool_ratio * 8.);
                    gt_geo_tail[2 * out_height_ * out_width_ + h * out_width_
                                + w] = (pool_ratio * w - points[quad[2]].x) / (pool_ratio * 8.);
                    gt_geo_tail[3 * out_height_ * out_width_ + h * out_width_
                                + w] = (h * pool_ratio - points[quad[2]].y) / (pool_ratio * 8.);
                    gt_geo_tail_weight[h * out_width_ + w] = weight_param;
                    gt_geo_tail_weight[out_height_ * out_width_ + h * out_width_ + w]     = weight_param;
                    gt_geo_tail_weight[2 * out_height_ * out_width_ + h * out_width_ + w] = weight_param;
                    gt_geo_tail_weight[3 * out_height_ * out_width_ + h * out_width_ + w] = weight_param;
                }
                else if (poly_mask_origin.at<float>(int(pool_ratio * h), int(pool_ratio * w)) == 4)
                {
                    // Set all pixels with 4 as top&bottom boundary region
                    gt_score_map[h * out_width_ + w] = 4;
                }
            }
        }
    }
    // Randomly mask background pixels to balance foreground pixels and background pixels.
    balance_BF(gt_score_map, ignore_ratio, gt_mask, out_height_, out_width_);
}
