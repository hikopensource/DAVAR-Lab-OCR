/*
##################################################################################################
// Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
// Filename       :    east_data.cpp
// Abstract       :    GT_mask generating in EAST

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

vector<vector<cv::Point> > Shrink_Poly(vector<vector<cv::Point> > poly, float R);
double distance_to_Line(cv::Point2f line_start, cv::Point2f line_end, cv::Point2f point);
void balance_BF(float* conf_data, float fg_ratio, int* mask_data, int out_height_, int out_width_);
extern "C" void parse_east_data(int height,
	int width,
	int* gt_boxes,
	int gt_boxes_size,
	int* gt_boxes_ignore,
	int gt_boxes_ignore_size,
	int pool_ratio,
	int geometry,  //0 RBOX 1 QUAD
	int label_shape,  // 0 normal 1 gaussian
	float shrink_ratio,
	float ignore_ratio,
	float* gt_score_map,
	int* gt_score_map_mask,
	float* gt_geo_data,
	float* gt_geo_weight,
    int seed
	);

vector<vector<cv::Point> > Shrink_Poly(vector<vector<cv::Point> > poly, float R)
{
    /*
        Shrink gt_bboxes according to factor R.
    */

	vector<vector<cv::Point> > tmp = poly;
	for (int poly_idx = 0; poly_idx < tmp.size(); ++poly_idx)
	{
		vector<cv::Point> p = tmp[poly_idx];
		float four_side[4] = { 0 };
		for (int p_idx = 0; p_idx < 4; ++p_idx)
		{
			float length1 = sqrt(pow(p[p_idx].x - p[p_idx + 1 > 3 ? 0 : p_idx + 1].x, 2) + pow(p[p_idx].y - p[p_idx + 1 > 3 ? 0 : p_idx + 1].y, 2));
			float length2 = sqrt(pow(p[p_idx].x - p[p_idx - 1 < 0 ? 3 : p_idx - 1].x, 2) + pow(p[p_idx].y - p[p_idx - 1 < 0 ? 3 : p_idx - 1].y, 2));
			four_side[p_idx] = MIN(length1, length2);
		}
		float _0_1 = sqrt(pow(p[0].x - p[1].x, 2) + pow(p[0].y - p[1].y, 2));
		float _2_3 = sqrt(pow(p[2].x - p[3].x, 2) + pow(p[2].y - p[3].y, 2));
		float _0_3 = sqrt(pow(p[0].x - p[3].x, 2) + pow(p[0].y - p[3].y, 2));
		float _1_2 = sqrt(pow(p[2].x - p[1].x, 2) + pow(p[2].y - p[1].y, 2));
		// Find longer side
		if (_0_1 + _2_3 > _1_2 + _0_3)
		{
			// p0, p1
			double theta = atan2(p[1].y - p[0].y, p[1].x - p[0].x);
			poly[poly_idx][0].x += R*four_side[0] * cos(theta);
			poly[poly_idx][0].y += R*four_side[0] * sin(theta);
			poly[poly_idx][1].x -= R*four_side[1] * cos(theta);
			poly[poly_idx][1].y -= R*four_side[1] * sin(theta);
			// p2, p3
			theta = atan2(p[2].y - p[3].y, p[2].x - p[3].x);
			poly[poly_idx][3].x += R*four_side[3] * cos(theta);
			poly[poly_idx][3].y += R*four_side[3] * sin(theta);
			poly[poly_idx][2].x -= R*four_side[2] * cos(theta);
			poly[poly_idx][2].y -= R*four_side[2] * sin(theta);
			// p0, p3
			theta = atan2(p[3].y - p[0].y, p[3].x - p[0].x);
			poly[poly_idx][0].x += R*four_side[0] * cos(theta);
			poly[poly_idx][0].y += R*four_side[0] * sin(theta);
			poly[poly_idx][3].x -= R*four_side[3] * cos(theta);
			poly[poly_idx][3].y -= R*four_side[3] * sin(theta);
			// p1, p2
			theta = atan2(p[1].y - p[2].y, p[1].x - p[2].x);
			poly[poly_idx][2].x += R*four_side[2] * cos(theta);
			poly[poly_idx][2].y += R*four_side[2] * sin(theta);
			poly[poly_idx][1].x -= R*four_side[1] * cos(theta);
			poly[poly_idx][1].y -= R*four_side[1] * sin(theta);
		}
		else
		{
			// p0, p3
			double theta = atan2(p[3].y - p[0].y, p[3].x - p[0].x);
			poly[poly_idx][0].x += R*four_side[0] * cos(theta);
			poly[poly_idx][0].y += R*four_side[0] * sin(theta);
			poly[poly_idx][3].x -= R*four_side[3] * cos(theta);
			poly[poly_idx][3].y -= R*four_side[3] * sin(theta);
			// p1, p2
			theta = atan2(p[1].y - p[2].y, p[1].x - p[2].x);
			poly[poly_idx][2].x += R*four_side[2] * cos(theta);
			poly[poly_idx][2].y += R*four_side[2] * sin(theta);
			poly[poly_idx][1].x -= R*four_side[1] * cos(theta);
			poly[poly_idx][1].y -= R*four_side[1] * sin(theta);
			// p0, p1
			theta = atan2(p[1].y - p[0].y, p[1].x - p[0].x);
			poly[poly_idx][0].x += R*four_side[0] * cos(theta);
			poly[poly_idx][0].y += R*four_side[0] * sin(theta);
			poly[poly_idx][1].x -= R*four_side[1] * cos(theta);
			poly[poly_idx][1].y -= R*four_side[1] * sin(theta);
			// p2, p3
			theta = atan2(p[2].y - p[3].y, p[2].x - p[3].x);
			poly[poly_idx][3].x += R*four_side[3] * cos(theta);
			poly[poly_idx][3].y += R*four_side[3] * sin(theta);
			poly[poly_idx][2].x -= R*four_side[2] * cos(theta);
			poly[poly_idx][2].y -= R*four_side[2] * sin(theta);
		}
	}
	return poly;
}

double distance_to_Line(cv::Point2f line_start, cv::Point2f line_end, cv::Point2f point)
{
    /*
        Calculate point-to-line distance
    */

	double normalLength = hypot(line_end.x - line_start.x, line_end.y - line_start.y);
	double distance = (double)((point.x - line_start.x) * (line_end.y - line_start.y) -
	                    (point.y - line_start.y) * (line_end.x - line_start.x)) / MAX(normalLength,0.0001);
	return abs(distance);
}

void balance_BF(float* conf_data, float fg_ratio, int* mask_data, int out_height_, int out_width_){
	/*
	    Random select foreground and background pixels
	    Args:
            conf_data: score_map of the segmentation
            fg_ratio: the ratio of foreground pixels / background pixels
            mask_data: the mask (weight) of score_map
            out_height_: height of score_map
            out_width_: width of score_map
	*/
	vector<int> fg_inds;
	fg_inds.clear();
	vector<int> bg_inds;
	bg_inds.clear();
	for (int h = 0; h < out_height_; h++) {
		for (int w = 0; w < out_width_; w++) {
			int index = h*out_width_ + w;
			if ((conf_data[index] - 0)<1e-5) {
				bg_inds.push_back(index);
			}
			else {
				fg_inds.push_back(index);
			}
		}
	}

	int num_fg_all = fg_inds.size();
	int num_bg_all = bg_inds.size();
	if (num_bg_all * fg_ratio > num_fg_all){
		random_shuffle(bg_inds.begin(), bg_inds.end());
		for (int i = 0; i < num_bg_all * fg_ratio - num_fg_all; i++) {
			if (mask_data != NULL){
				mask_data[bg_inds[i]] = 0;
			}
		}
	}
}
	

extern "C" void parse_east_data(int height,
								int width,
								int* gt_boxes,
								int gt_boxes_size,
								int* gt_boxes_ignore,
								int gt_boxes_ignore_size,
								int pool_ratio,
								int geometry,  //0 RBOX 1 QUAD
								int label_shape,  // 0 normal 1 gaussian
								float shrink_ratio,
								float ignore_ratio,
								float* gt_score_map,
								int* gt_score_map_mask,
								float* gt_geo_data,
								float* gt_geo_weight,
                                int seed)
{
    srand(seed);
	int out_height_ = height / pool_ratio;
	int out_width_ = width / pool_ratio;

	cv::Mat conf_origin = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat geo_origin;
	cv::Mat geo_mask_origin;
	cv::Mat temp_N_Q_data;

	// Output Initialization
	for (int i = 0; i < out_height_ * out_width_; i++){
		gt_score_map[i] = 0;
		gt_score_map_mask[i] = 1;
	}

	if (geometry == 0)
	{
	    //RBOX
		for (int i = 0; i < 5 * out_height_ * out_width_; i++){
			gt_geo_data[i] = 0;
			gt_geo_weight[i] = 0;
		}
	}
	else if (geometry == 1)
	{
	    //QUAD
		for (int i = 0; i < 8 * out_height_ * out_width_; i++){
			gt_geo_data[i] = 0;
			gt_geo_weight[i] = 0;
		}
	}

    // Get ignore bboxes, fill all ignore boxes with label 64 and set gt_score_map_mask to 0
	vector<vector<cv::Point> > poly_ignore;
	poly_ignore.clear();
	for (int i = 0; i < gt_boxes_ignore_size; i++) 
	{
		vector<cv::Point> poly_tmp;
		for (int j = 0; j<8; j += 2)
		{
			poly_tmp.push_back(cv::Point(gt_boxes_ignore[i * 8 + j], gt_boxes_ignore[i * 8 + j + 1]));
		}
		poly_ignore.push_back(poly_tmp);
	}
	cv::fillPoly(conf_origin, poly_ignore, 64);
	for (int h = 0; h < out_height_; h++) 
	{
		for (int w = 0; w < out_width_; w++) 
		{
			if (conf_origin.at<float>(int(pool_ratio*h), int(pool_ratio*w)) == 64)
			{
				gt_score_map_mask[h*out_width_ + w] = 0;
			}
		}
	}

    // Get cared bboxes
	vector<vector<cv::Point> > poly;
	poly.clear();
	for (int i = 0; i < gt_boxes_size; i++) {
		vector<cv::Point> poly_tmp;
		for (int j = 0; j < 8; j += 2){
			poly_tmp.push_back(cv::Point(gt_boxes[i * 8 + j], gt_boxes[i * 8 + j + 1]));
		}
		poly.push_back(poly_tmp);
	}

	// Shrink bboxes
	vector<vector<cv::Point> > shrinked_poly = Shrink_Poly(poly, shrink_ratio);

    // Use conf_origin == 1 to mark cared bboxes
    //cv::fillPoly(conf_origin, shrinked_poly, 1);

    //
    for (int poly_idx = 0; poly_idx < poly.size(); ++poly_idx)
	{

		vector<cv::Point> point_tmp = poly[poly_idx];

        // Use conf_origin_tmp == 1 to mark cared bboxes
        cv::Mat conf_origin_tmp = cv::Mat::zeros(height, width, CV_32FC1);
		vector<vector<cv::Point> > shrink_poly_tmps;
        shrink_poly_tmps.push_back(shrinked_poly[poly_idx]);
        cv::fillPoly(conf_origin_tmp, shrink_poly_tmps, 1);

        float center_x, center_y, sigma;
        if (label_shape == 1)
		{
		    // If generate score map in Gaussian mode, we need to calculate the center point and sigma
			vector<cv::Point> shrink_poly_tmp = shrinked_poly[poly_idx];
			float shrinked_four_side[4] = { 0 };
			for (int p_idx = 0; p_idx < 4; ++p_idx)
			{
				shrinked_four_side[p_idx] = sqrt(pow(shrink_poly_tmp[p_idx].x - shrink_poly_tmp[p_idx + 1 > 3 ? 0 : p_idx + 1].x, 2)
				                 + pow(shrink_poly_tmp[p_idx].y - shrink_poly_tmp[p_idx + 1 > 3 ? 0 : p_idx + 1].y, 2));;
			}
			float shrink_poly_h = MAX(shrinked_four_side[1], shrinked_four_side[3]);
			float shrink_poly_w = MAX(shrinked_four_side[0], shrinked_four_side[2]);

			sigma = MAX(shrink_poly_h, shrink_poly_w) / 2 * 0.7485;
			center_x = (shrink_poly_tmp[0].x + shrink_poly_tmp[1].x + shrink_poly_tmp[2].x + shrink_poly_tmp[3].x) / 4.;
			center_y = (shrink_poly_tmp[0].y + shrink_poly_tmp[1].y + shrink_poly_tmp[2].y + shrink_poly_tmp[3].y) / 4.;
		}

		if (geometry == 0)
		{
		    /********** RBOX **********/
			cv::RotatedRect rbox = cv::minAreaRect(point_tmp);
			cv::Point2f vertices[4];
			rbox.points(vertices);
			float angle = rbox.angle;

			// Switch order
			cv::Point2f min_area_box[4];
			if (abs(angle) > 45.0)
			{
				angle = -(90.0 + angle) / 180.0*3.1415926535897;
				min_area_box[0] = vertices[2];
				min_area_box[1] = vertices[3];
				min_area_box[2] = vertices[0];
				min_area_box[3] = vertices[1];
			}
			else
			{
				angle = abs(angle) / 180.0*3.1415926535897;
				min_area_box[0] = vertices[1];
				min_area_box[1] = vertices[2];
				min_area_box[2] = vertices[3];
				min_area_box[3] = vertices[0];
			}

            // Generate final gt target for each box
			for (int h = 0; h < out_height_; ++h)
			{
				for (int w = 0; w < out_width_; ++w)
				{
					if (conf_origin_tmp.at<float>(h * pool_ratio, w * pool_ratio) == 1)
					{
					     // generate score map for each bbox
						if (label_shape == 1)
						{
						    // if using gaussian score map
							gt_score_map[h*out_width_ + w] = exp(-0.5 / pow(sigma, 2) * (pow(h*pool_ratio - center_y, 2) + pow(w*pool_ratio - center_x, 2)));
						}
						else
						{
						    // if using normal score map
						    gt_score_map[h*out_width_ + w] = 1;
						}

                        // generate geo map for each bbox
						gt_geo_data[0*out_height_*out_width_ + h*out_width_ + w] = (float)distance_to_Line(min_area_box[0], min_area_box[1], cv::Point2f(w * pool_ratio, h * pool_ratio));
						gt_geo_data[1*out_height_*out_width_ + h*out_width_ + w] = (float)distance_to_Line(min_area_box[1], min_area_box[2], cv::Point2f(w * pool_ratio, h * pool_ratio));
						gt_geo_data[2*out_height_*out_width_ + h*out_width_ + w] = (float)distance_to_Line(min_area_box[2], min_area_box[3], cv::Point2f(w * pool_ratio, h * pool_ratio));
						gt_geo_data[3*out_height_*out_width_ + h*out_width_ + w] = (float)distance_to_Line(min_area_box[3], min_area_box[0], cv::Point2f(w * pool_ratio, h * pool_ratio));
						gt_geo_data[4*out_height_*out_width_ + h*out_width_ + w] = (float)angle;

						for (int k = 0; k < 5; k++)
						{
							gt_geo_weight[k*out_height_*out_width_ + h*out_width_ + w] = 1;
						}

					}
				}
			}

		}
	    else if (geometry == 1)
	    {
	        /********** QUAD **********/
            float four_side[4] = { 0 };
			for (int p_idx = 0; p_idx < 4; ++p_idx)
			{
				four_side[p_idx] = sqrt(pow(point_tmp[p_idx].x - point_tmp[(p_idx + 1) % 4].x, 2)
				                +  pow(point_tmp[p_idx].y - point_tmp[(p_idx + 1)%4].y, 2));
			}

			 // Generate final gt target for each box
			for (int h = 0; h < out_height_; ++h)
			{
				for (int w = 0; w < out_width_; ++w)
				{
					if (conf_origin_tmp.at<float>(h * pool_ratio, w * pool_ratio) == 1)
					{
					     // generate score map for each bbox
						if (label_shape == 1)
						{
						    // if using gaussian score map
							gt_score_map[h*out_width_ + w] = exp(-0.5 / pow(sigma, 2) * (pow(h*pool_ratio - center_y, 2) + pow(w*pool_ratio - center_x, 2)));
						}
						else
						{
						    // if using normal score map
						    gt_score_map[h*out_width_ + w] = 1;
						}

                        // generate geo map for each bbox
                        for (int k = 0; k < 4; ++k )
                        {
                            gt_geo_data[2*k*out_height_*out_width_ + h*out_width_ + w] = point_tmp[k].x - w*pool_ratio;
                            gt_geo_data[(2*k+1)*out_height_*out_width_ + h*out_width_ + w] = point_tmp[k].y - h*pool_ratio;

                            // Balance large text and small text according to their side length
                            float min_side = four_side[k]<four_side[(k + 1) % 4] ? four_side[k] : four_side[(k + 1) % 4];
                            gt_geo_weight[2*k*out_height_*out_width_ + h*out_width_ + w] = 1./(min_side + 1e-7);
                            gt_geo_weight[(2*k+1)*out_height_*out_width_ + h*out_width_ + w] = 1./(min_side + 1e-7);
                        }
					}
				}
			}
	    }
	}
	balance_BF(gt_score_map, ignore_ratio, gt_score_map_mask, out_height_, out_width_);
}




	



 
