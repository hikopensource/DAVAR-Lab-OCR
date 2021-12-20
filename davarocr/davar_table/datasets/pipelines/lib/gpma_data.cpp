/*
##################################################################################################
// Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
// Filename       :    gpma_data.cpp
// Abstract       :    generating gt_mask for GPMA branch training in LGPMA

// Current Version:    1.0.0
// Date           :    2021-09-18
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

void balance_BF(int* conf_data, float fg_ratio, int* mask_data, int out_height_, int out_width_);
extern "C" void generate_seg_data(int height ,
                                      int width,
                                      float* gt_boxes,
                                      int gt_boxes_size,
                                      float* shrink_gt_boxes,
                                      int shrink_gt_boxes_length,
                                       float* empty_gt_boxes,
                                      int empty_gt_boxes_length,
                                      float* small_gt_bboxes,
                                      int small_gt_bboxes_size,
                                      int pool_ratio,
                                      float ignore_ratio,
                                      int* gt_score_map,
                                      int* gt_score_map_weight,
                                      float* gt_geo_bond,
                                      float* gt_geo_bond_weight);



void balance_BF(int  *conf_data,
                float fg_ratio,
                int  *mask_data,
                int   out_height_,
                int   out_width_)
{
    /*
        Description:
            Balance foreground and background pixels.
            Set some of the background pixels' weight to 0 according to the fg_ratio
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
     if (num_fg_all * fg_ratio > num_bg_all)
    {
        // Randomly select background pixels
        random_shuffle(fg_inds.begin(), fg_inds.end());
        for (int i = 0; i < num_fg_all * fg_ratio - num_bg_all; i++)
        {
            if (mask_data != NULL)
            {
                mask_data[fg_inds[i]] = 0;
            }
        }
    }
}

extern "C" void generate_seg_data (int height ,
                                      int width,
                                      float* gt_boxes,
                                      int gt_boxes_size,
                                      float* shrink_gt_boxes,
                                      int shrink_gt_boxes_length,
                                      float* empty_gt_boxes,
                                      int empty_gt_boxes_length,
                                      float* small_gt_bboxes,
                                      int small_gt_bboxes_size,
                                      int pool_ratio,
                                      float ignore_ratio,
                                      int* gt_score_map,
                                      int* gt_score_map_weight,
                                      float* gt_geo_bond,
                                      float* gt_geo_bond_weight)
    /*
        Description:
            The function that exposed to python.
            According to the annotations of polygon points, generate the segmentation score_map and regression geo-map
        Args:
            height: height of the original image, int.
            width: width of the original image, int
            gt_boxes: the ground_thruth of polygon points [[x1,y1,x2,y2],...]
            gt_boxes_size: the length of gt_boxes, int
            empty_gt_boxes: the ground_thruth of empty points [[x1,y1,x2,y2],...]
            empty_gt_boxes_length: the length of empty_gt_bboxes, int
            small_gt_bboxes, original bboxes  [[x1,y1,x2,y2],...]
            small_gt_bboxes_size, original bboxes size
            pool_ratio: the downsample ratio of feature map versus original image.
            ignore_ratio: parameter to control the ratio of foreground pixels and background pixels.

        Return:
            gt_score_map: the segmentation score map [H, W]
            gt_score_map_weight:  the mask(weight) of segmentation score map [H, W]
            gt_geo_bond: the regression score map of top&bottom boundary region [2, H, W]
            gt_geo_bond_weight: the mask(weight) of regression map of top&bottom boundary region, [2, H, W]
     */
{
    // Initialize an empty mat
    vector<vector<cv::Point> > draw_tmp;
    cv::Mat                    poly_mask_origin_text = cv::Mat::zeros(height, width, CV_32FC1);
    int                        out_height_           = height / pool_ratio; // feature map height
    int                        out_width_            = width / pool_ratio;  // feature map width

    // Initialize the return value
    for (int i = 0; i < out_height_ * out_width_; i++)
    {
        gt_score_map[i] = 0;
        gt_score_map_weight[i]      = 1;
    }
    for (int i = 0; i < 2 * out_height_ * out_width_; i++)
    {
        gt_geo_bond[i]        = 0;
        gt_geo_bond_weight[i] = 0;
    }

    // Draw all of the empty polygon regions with 1
    draw_tmp.clear();
     for (int i = 0; i < empty_gt_boxes_length; i++)
    {
        vector<cv::Point> poly;
        poly.push_back(cv::Point(empty_gt_boxes[i * 4], empty_gt_boxes[i * 4 + 1]));
        poly.push_back(cv::Point(empty_gt_boxes[i * 4 + 2], empty_gt_boxes[i * 4 + 1]));
        poly.push_back(cv::Point(empty_gt_boxes[i * 4 + 2], empty_gt_boxes[i * 4 + 3]));
        poly.push_back(cv::Point(empty_gt_boxes[i * 4], empty_gt_boxes[i * 4 + 3]));
        draw_tmp.push_back(poly);
    }
    cv::fillPoly(poly_mask_origin_text, draw_tmp, 1);
     for (int i = 0; i < shrink_gt_boxes_length; i++)
    {
        vector<cv::Point> poly;
        poly.push_back(cv::Point(shrink_gt_boxes[i * 4], shrink_gt_boxes[i * 4 + 1]));
        poly.push_back(cv::Point(shrink_gt_boxes[i * 4 + 2], shrink_gt_boxes[i * 4 + 1]));
        poly.push_back(cv::Point(shrink_gt_boxes[i * 4 + 2], shrink_gt_boxes[i * 4 + 3]));
        poly.push_back(cv::Point(shrink_gt_boxes[i * 4], shrink_gt_boxes[i * 4 + 3]));
        draw_tmp.push_back(poly);
    }
    cv::fillPoly(poly_mask_origin_text, draw_tmp, 1);
    for (int h = 0; h < out_height_; h++) {
		for (int w = 0; w < out_width_; w++) {
		    if (poly_mask_origin_text.at<float>(int(pool_ratio*h), int(pool_ratio*w)) == 1) {
		        gt_score_map[h*out_width_ + w] = 1;
		    }
		}
	}


	// Draw all of the cared polygon regions with 1.
	for(int i = 0; i < gt_boxes_size; i++)
	{
	    float middle_point_x = (small_gt_bboxes[i * 4 + 2] + small_gt_bboxes[i*4]) / 2;
	    float middle_point_y = (small_gt_bboxes[i * 4 + 3] + small_gt_bboxes[i*4+1]) / 2;

        // Set the weight param to prevent the L1 loss become too big.
		float weight_param = (pool_ratio * pool_ratio)/(256.);

		float x1 = gt_boxes[i * 4];
		float y1 = gt_boxes[i * 4 + 1];
		float x2 = gt_boxes[i * 4 + 2];
		float y2 = gt_boxes[i * 4 + 3];

		for (int h = 0; h < out_height_; h++) {
			for (int w = 0; w < out_width_; w++) {
				 if( (pool_ratio*h) >= y1 && (pool_ratio*h) <= y2 && (pool_ratio *w) >= x1 && (pool_ratio*w) <= x2){

					 // Generate the center text to boundaries offset.
					 if (middle_point_x != gt_boxes[i*4] && middle_point_x != gt_boxes[i*4+2]){
					    if (w * pool_ratio <= middle_point_x){
					     gt_geo_bond[h*out_width_ + w] =
					        (w * pool_ratio - gt_boxes[i * 4])/(middle_point_x - gt_boxes[i*4]);
                         }
                         else{
                             gt_geo_bond[h*out_width_ + w] =
                                (w * pool_ratio - gt_boxes[i * 4 + 2]) / (middle_point_x - gt_boxes[i*4 + 2]);
                         }
                          gt_geo_bond_weight[h*out_width_ + w] += weight_param;
					 }


                      if (middle_point_y != gt_boxes[i*4+1] && middle_point_y != gt_boxes[i*4+3]){
                           if (h * pool_ratio <= middle_point_y){
                                gt_geo_bond[1*out_height_*out_width_ + h*out_width_ + w] =
                                    (h * pool_ratio - gt_boxes[i * 4 + 1])/(middle_point_y - gt_boxes[i*4 + 1]);
                             }else{
                                 gt_geo_bond[1*out_height_*out_width_ + h*out_width_ + w] =
                                    (h * pool_ratio - gt_boxes[i * 4 + 3])/(middle_point_y - gt_boxes[i*4 + 3]);
                             }
                         gt_geo_bond_weight[1*out_height_*out_width_ + h*out_width_ + w] += weight_param;
                    }



				}
			}
		}
	}
	float weight_param = (pool_ratio * pool_ratio)/(256.);
	for (int i = 0; i < 2 * out_height_ * out_width_; i++)
    {
        if (gt_geo_bond_weight[i] > weight_param){
            gt_geo_bond_weight[i] = 0;
        }
    }
	 // Randomly mask background pixels to balance foreground pixels and background pixels.
	balance_BF(gt_score_map, ignore_ratio, gt_score_map_weight, out_height_, out_width_);
}

