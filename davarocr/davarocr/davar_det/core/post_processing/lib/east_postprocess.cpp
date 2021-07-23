/*
 ##################################################################################################
   // Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
   // Filename       :    east_postporcess.cpp
   // Abstract       :    Generating predict boxes for EAST

   // Current Version:    1.0.0
   // Date           :    2020-05-31
 ###################################################################################################
 */
#include <opencv2/opencv.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream> // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

// ================================== Macro Definition ====================================
#define EAST_OUT_PRECISION      10000           // Used to scale box for precisely NMS
#define EAST_OUT_OUTPUT_FORM    9               // output number for each instance: x1 y1 x2 y2 x3 y3 x4 y4 conf
#define EAST_OUT_POS_FORM       8               // coordinate number: x1 y1 x2 y2 x3 y3 x4 y4
#define EAST_MAX_BOX_NUM        500             // Maximum output boxes
#define EAST_OUT_CACHE_SIZE     25              // alloc cache size
#define SUPPRESSED              -1.0f           // suppressed text with confidence -1
// ================================== Struct Definition ====================================
typedef struct _EAST_POINT_
{
    float x;
    float y;
} EAST_POINT;

typedef struct _EAST_POLY_
{
    EAST_POINT point[4];
} EAST_POLY;

typedef struct _EAST_POLYGON_
{
    EAST_POLY poly;
    float     score;
} EAST_POLYGON;

// ================================== Function Definition ====================================
void EAST_reverse(EAST_POLY *poly1);
float EAST_OUT_area(EAST_POLY *poly1,
                    int        n);
int EAST_sig(float d);
float EAST_cross(EAST_POINT *o,
                 EAST_POINT *a,
                 EAST_POINT *b);
int EAST_lineCross(EAST_POINT *a,
                   EAST_POINT *b,
                   EAST_POINT *c,
                   EAST_POINT *d,
                   EAST_POINT *p);
int EAST_polygon_cut(EAST_POINT *p,
                     int         n,
                     EAST_POINT *a,
                     EAST_POINT *b);
void EAST_swap(EAST_POINT *a,
               EAST_POINT *b);
float EAST_OUT_intersect_area(EAST_POINT *pa,
                              EAST_POINT *pb,
                              EAST_POINT *pc,
                              EAST_POINT *pd);
float EAST_OUT_box_intersection(EAST_POLYGON *poly1,
                                EAST_POLYGON *poly2);
float EAST_OUT_box_iou(EAST_POLYGON *poly1,
                       EAST_POLYGON *poly2);
void EAST_OUT_sort(float *pos_data,
                   float *conf_data,
                   int    box_num);
int EAST_OUT_should_merge(EAST_POLYGON *poly1,
                          EAST_POLYGON *poly2,
                          float         iou_threshold);
void EAST_OUT_merge(EAST_POLYGON *poly1,
                    EAST_POLYGON *poly2);
void EAST_OUT_copy_polygon(float        *pos_data,
                           float        *conf_data,
                           int           id,
                           EAST_POLYGON *src);
int EAST_OUT_lanms(float *pos_data,
                   float *conf_data,
                   int    n,
                   float  iou_threshold);
int EAST_OUT_nms(float *pos_data,
                 float *conf_data,
                 float  nms_thresh,
                 int    box_num);


extern "C" void generate_result(float *score_map,
                                float *geo_map,
                                int    height,
                                int    width,
                                int    pool_ratio,
                                float  scale_factor,
                                float  thresh_text,
                                float  nms_thresh,
                                int    nms_method,
                                float *result,
                                int   *result_num);

// ================================= Function Implement ==================================

void EAST_reverse(EAST_POLY *poly1)

/*
    Change points order from 0,1,2,3 into 3,2,1,0
 */
{
    EAST_POINT temp_point;

    // swap 0,3
    temp_point.x      = poly1->point[3].x;
    temp_point.y      = poly1->point[3].y;
    poly1->point[3].x = poly1->point[0].x;
    poly1->point[3].y = poly1->point[0].y;
    poly1->point[0].x = temp_point.x;
    poly1->point[0].y = temp_point.y;

    // swap 1,2
    temp_point.x      = poly1->point[1].x;
    temp_point.y      = poly1->point[1].y;
    poly1->point[1].x = poly1->point[2].x;
    poly1->point[1].y = poly1->point[2].y;
    poly1->point[2].x = temp_point.x;
    poly1->point[2].y = temp_point.y;
}

float EAST_OUT_area(EAST_POLY *poly1,
                    int        n)
{
    int        i   = 0;
    float      res = 0;
    EAST_POLY *ps  = poly1;

    for (i = 0; i < n; i++)
    {
        res += ps->point[i].x * ps->point[(i + 1) % n].y - ps->point[i].y * ps->point[(i + 1) % n].x;
    }
    return res / 2.0f;
}

int EAST_sig(float d)
{
    const float eps = 1E-8f;

    return (d > eps) - (d < -eps);
}

float EAST_cross(EAST_POINT *o,
                 EAST_POINT *a,
                 EAST_POINT *b)
{
    return (a->x - o->x) * (b->y - o->y) - (b->x - o->x) * (a->y - o->y);
}

int EAST_lineCross(EAST_POINT *a,
                   EAST_POINT *b,
                   EAST_POINT *c,
                   EAST_POINT *d,
                   EAST_POINT *p)
{
    float s1, s2;

    s1 = EAST_cross(a, b, c);
    s2 = EAST_cross(a, b, d);
    if ((EAST_sig(s1) == 0) && (EAST_sig(s2) == 0))
    {
        return 2;
    }
    if (EAST_sig(s2 - s1) == 0)
    {
        return 0;
    }
    p->x = (c->x * s2 - d->x * s1) / (s2 - s1);
    p->y = (c->y * s2 - d->y * s1) / (s2 - s1);
    return 1;
}

int EAST_polygon_cut(EAST_POINT *p,
                     int         n,
                     EAST_POINT *a,
                     EAST_POINT *b)
{
    EAST_POINT pp[10];
    int        m = 0;
    int        i = 0;

    p[n] = p[0];
    for (i = 0; i < n; i++)
    {
        if (EAST_sig(EAST_cross(a, b, &p[i])) > 0)
        {
            pp[m++] = p[i];
        }
        if (EAST_sig(EAST_cross(a, b, &p[i])) != EAST_sig(EAST_cross(a, b, &p[i + 1])))
        {
            EAST_lineCross(a, b, &p[i], &p[i + 1], &pp[m++]);
        }
    }
    n = 0;
    for (i = 0; i < m; i++)
    {
        if (!i || !((EAST_sig(pp[i].x - pp[i - 1].x) == 0) && (EAST_sig(pp[i].y - pp[i - 1].y) == 0)))
        {
            p[n++] = pp[i];
        }
    }

    while (n > 1 && (EAST_sig(p[n - 1].x - p[0].x) == 0 && EAST_sig(p[n - 1].y - p[0].y) == 0))
    {
        n--;
    }
    return n;
}

void EAST_swap(EAST_POINT *a,
               EAST_POINT *b)
{
    float temp_x = 0;
    float temp_y = 0;

    temp_x = a->x;
    temp_y = a->y;
    a->x   = b->x;
    a->y   = b->y;
    b->x   = temp_x;
    b->y   = temp_y;
}

float EAST_OUT_intersect_area(EAST_POINT *pa,
                              EAST_POINT *pb,
                              EAST_POINT *pc,
                              EAST_POINT *pd)
{
    EAST_POINT o;
    EAST_POINT p[10];
    EAST_POINT a, b, c, d;

    int   s1  = 0;
    int   s2  = 0;
    int   n   = 3;
    float res = 0;

    o.x = 0;
    o.y = 0;
    memcpy(&a, pa, sizeof(EAST_POINT));
    memcpy(&b, pb, sizeof(EAST_POINT));
    memcpy(&c, pc, sizeof(EAST_POINT));
    memcpy(&d, pd, sizeof(EAST_POINT));

    s1 = EAST_sig(EAST_cross(&o, &a, &b));
    s2 = EAST_sig(EAST_cross(&o, &c, &d));
    if ((s1 == 0) || (s2 == 0))
    {
        return 0.0;
    }
    if (s1 == -1)
    {
        EAST_swap(&a, &b);
    }
    if (s2 == -1)
    {
        EAST_swap(&c, &d);
    }
    memcpy(&p[0], &o, sizeof(EAST_POINT));
    memcpy(&p[1], &a, sizeof(EAST_POINT));
    memcpy(&p[2], &b, sizeof(EAST_POINT));

    n   = EAST_polygon_cut(p, n, &o, &c);
    n   = EAST_polygon_cut(p, n, &c, &d);
    n   = EAST_polygon_cut(p, n, &d, &o);
    res = fabs(EAST_OUT_area((EAST_POLY *)p, n));
    if (s1 * s2 == -1)
    {
        res = -res;
    }
    return res;
}

float EAST_OUT_box_intersection(EAST_POLYGON *poly1,
                                EAST_POLYGON *poly2)
{
    float        res = 0;
    int          i, j;
    EAST_POLYGON poly_temp1;
    EAST_POLYGON poly_temp2;

    memcpy(&poly_temp1, poly1, sizeof(EAST_POLYGON));
    memcpy(&poly_temp2, poly2, sizeof(EAST_POLYGON));
    EAST_POLY *poly_1 = &poly_temp1.poly;
    EAST_POLY *poly_2 = &poly_temp2.poly;
    if (EAST_OUT_area(poly_1, 4) < 0)
    {
        EAST_reverse(poly_1);
    }
    if (EAST_OUT_area(poly_2, 4) < 0)
    {
        EAST_reverse(poly_2);
    }
    for (i = 0; i < 4; i++)
    {
        for (j = 0; j < 4; j++)
        {
            res +=
                EAST_OUT_intersect_area(&(poly_1->point[i]), &(poly_1->point[(i + 1) % 4]), &(poly_2->point[j]),
                                        &(poly_2->point[(j + 1) % 4]));
        }
    }
    return res;
}

float EAST_OUT_box_iou(EAST_POLYGON *poly1,
                       EAST_POLYGON *poly2)
{
    float box_intersection = 0;
    float box_union        = 0;

    box_intersection = EAST_OUT_box_intersection(poly1, poly2);
    box_union        = fabs(EAST_OUT_area(&poly1->poly, 4)) + fabs(EAST_OUT_area(&poly2->poly, 4)) - box_intersection;
    return box_intersection / box_union;
}


// Sort detected boxes by their confidences before NMS
void EAST_OUT_sort(float *pos_data,
                   float *conf_data,
                   int    box_num)
{
    int   i, j;
    float tmp_pos_data[8];
    float tmp_conf_data;

    for (i = 0; i < box_num; i++)
    {
        for (j = i + 1; j < box_num; j++)
        {
            if (conf_data[i] < conf_data[j])
            {
                memcpy(tmp_pos_data, &(pos_data[i * 8]), sizeof(float) * 8);
                memcpy(&(pos_data[i * 8]), &(pos_data[j * 8]), sizeof(float) * 8);
                memcpy(&(pos_data[j * 8]), tmp_pos_data, sizeof(float) * 8);

                memcpy(&tmp_conf_data, &conf_data[i], sizeof(float));
                conf_data[i] = conf_data[j];
                conf_data[j] = tmp_conf_data;
            }
        }
    }
}

// Judge whether two boxes can be merged
int EAST_OUT_should_merge(EAST_POLYGON *poly1,
                          EAST_POLYGON *poly2,
                          float         iou_threshold)
{
    return EAST_OUT_box_iou(poly1, poly2) > iou_threshold;
}

// Weighted merge
void EAST_OUT_merge(EAST_POLYGON *poly1,
                    EAST_POLYGON *poly2)
{
    // merge poly1&poly2 to poly2
    float      score_inv = 0;
    float      score1, score2;
    EAST_POLY *p1;
    EAST_POLY *p2;
    int        i, j;

    score1    = poly1->score;
    score2    = poly2->score;
    score_inv = 1.0f / MAX(1e-8f, score1 + score2);

    p1 = &poly1->poly;
    p2 = &poly2->poly;
    for (i = 0; i < 4; i++)
    {
        p2->point[i].x = (p1->point[i].x * score1 + p2->point[i].x * score2) * score_inv;
        p2->point[i].y = (p1->point[i].y * score1 + p2->point[i].y * score2) * score_inv;
    }
    poly2->score += poly1->score;
}

// Copy last polygon to output
void EAST_OUT_copy_polygon(float        *pos_data,
                           float        *conf_data,
                           int           id,
                           EAST_POLYGON *src)
{
    pos_data[id * 8]     = src->poly.point[0].x;
    pos_data[id * 8 + 1] = src->poly.point[0].y;
    pos_data[id * 8 + 2] = src->poly.point[1].x;
    pos_data[id * 8 + 3] = src->poly.point[1].y;
    pos_data[id * 8 + 4] = src->poly.point[2].x;
    pos_data[id * 8 + 5] = src->poly.point[2].y;
    pos_data[id * 8 + 6] = src->poly.point[3].x;
    pos_data[id * 8 + 7] = src->poly.point[3].y;
    conf_data[id]        = src->score;
}

// Implementation of lanms
int EAST_OUT_lanms(float *pos_data,
                   float *conf_data,
                   int    n,
                   float  iou_threshold)
{

    int    lanms_box_num = 0;      // Used to store final box numbers
    float *p             = NULL;

    int          i = 0, j = 0;
    EAST_POLYGON polygon;
    EAST_POLYGON last_polygon;

    for (i = 0; i < n; i++)
    {
        p = pos_data + i * 8;

        for (j = 0; j < 4; j++)
        {
            polygon.poly.point[j].x = p[j * 2];
            polygon.poly.point[j].y = p[j * 2 + 1];
        }
        polygon.score = conf_data[i];

        if (i == 0)
        {
            // the first one
            memcpy(&last_polygon, &polygon, sizeof(EAST_POLYGON));
            lanms_box_num++;
        }
        else
        {
            // merged with last one
            if (EAST_OUT_should_merge(&polygon, &last_polygon, iou_threshold))
            {
                // merge to the last_polygon
                EAST_OUT_merge(&polygon, &last_polygon);
            }
            else
            {
                // add merged box num
                lanms_box_num++;
                // update current polygon as last_ploygon
                memcpy(&last_polygon, &polygon, sizeof(EAST_POLYGON));
            }
        }
        EAST_OUT_copy_polygon(pos_data, conf_data, lanms_box_num - 1, &last_polygon);
    }
    return lanms_box_num;
}

// Standard NMS
int EAST_OUT_nms(float *pos_data,
                 float *conf_data,
                 float  nms_thresh,
                 int    box_num)
{
    int          i             = 0, j = 0, k = 0;
    int          valid_box_num = 0;
    EAST_POLYGON poly1, poly2;
    float        overlap = 0;
    float       *p       = NULL;

    EAST_OUT_sort(pos_data, conf_data, box_num);

    for (i = 0; i < box_num; i++)
    {
        p = pos_data + i * 8;
        for (k = 0; k < 4; k++)
        {
            poly1.poly.point[k].x = p[k * 2];
            poly1.poly.point[k].y = p[k * 2 + 1];
        }

        // If instance i is suppressed, then it won't be participant in nms
        if (conf_data[i] < -0.5f)
        {
            continue;
        }
        for (j = i + 1; j < box_num; j++)
        {
            p = pos_data + j * 8;
            for (k = 0; k < 4; k++)
            {
                poly2.poly.point[k].x = p[k * 2];
                poly2.poly.point[k].y = p[k * 2 + 1];
            }
            overlap = EAST_OUT_box_iou(&poly1, &poly2);

            if (overlap > nms_thresh)
            {
                conf_data[j] = SUPPRESSED;
            }
        }
    }

    j = 0;
    for (i = 0; i < box_num; i++)
    {
        if (conf_data[i] > -0.5f)
        {
            conf_data[j] = conf_data[i];

            pos_data[j * 8 + 0] = pos_data[i * 8 + 0] / EAST_OUT_PRECISION;
            pos_data[j * 8 + 1] = pos_data[i * 8 + 1] / EAST_OUT_PRECISION;
            pos_data[j * 8 + 2] = pos_data[i * 8 + 2] / EAST_OUT_PRECISION;
            pos_data[j * 8 + 3] = pos_data[i * 8 + 3] / EAST_OUT_PRECISION;
            pos_data[j * 8 + 4] = pos_data[i * 8 + 4] / EAST_OUT_PRECISION;
            pos_data[j * 8 + 5] = pos_data[i * 8 + 5] / EAST_OUT_PRECISION;
            pos_data[j * 8 + 6] = pos_data[i * 8 + 6] / EAST_OUT_PRECISION;
            pos_data[j * 8 + 7] = pos_data[i * 8 + 7] / EAST_OUT_PRECISION;
            j++;
        }
    }
    valid_box_num = j;
    return valid_box_num;
}



extern "C" void generate_result(float *score_map,
                                float *geo_map,
                                int    height,
                                int    width,
                                int    pool_ratio,
                                float  scale_factor,
                                float  thresh_text,
                                float  nms_thresh,
                                int    nms_method,
                                float *result,
                                int   *result_num)
/* Main function for east

Args:
    score_map: predicted score map, in shape of B*1*H*W
    geo_map: predicted geo map, in shape of B*5*H*W or B*8*H*W
    height: feature map height
    width: feature map width
    pool_ratio: feature map pooling ratio, default as 4.
    scale_factor: scale factor used to rescale result into original scale.
    thresh_text: threshold for binary classification text/background
    nms_thresh: threshold when doing nms
    nms_method: nms mode, 0: "RBOX', 1: "QUAD"
    result: used to store returned results
    result_num: used to store results numbers
*/
{
    int c             = 0;
    int count         = 0;
    int valid_box_num_all = 0;      // Used to store detected box numbers

    // Feature map size
    int feat_size = width * height;

    // Alloc some caches for following computation
    float *cache_data      = (float *)malloc(sizeof(float) * EAST_OUT_CACHE_SIZE * feat_size);
    float *cache_data_copy = cache_data;

    float *pos_data_out   = NULL; // Used to store box coordinate: x1,y1,x2,y2,x3,y3,x4,y4
    float *conf_data_out  = NULL; // Used to store confidence: conf
    float *res_merge      = NULL; // Used to store merged results
    float *pos_cur_out    = NULL; // Used to store final current position
    float *conf_cur_out   = NULL; //  Used to store final current confidence

    // Initialize the pointers
    pos_data_out   = cache_data;   // x1, y1, ..., x4, y,4
    cache_data    += feat_size * EAST_OUT_POS_FORM;
    conf_data_out  = cache_data;  // conf_map
    cache_data    += feat_size;
    res_merge      = cache_data; // x1, y1, ..., x4, y4

    pos_cur_out   = pos_data_out;   // Used to store current detected box start pointer
    conf_cur_out  = conf_data_out;  // Used to store current confidence pointer

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            float score = score_map[c * height * width + h * width + w];

            if (score > float(thresh_text))
            {

                if (nms_method == 1)
                {
                    // Quad mode
                    pos_cur_out[count * 8]     = pool_ratio * w + geo_map[h * width + w];                 // x1
                    pos_cur_out[count * 8 + 1] = pool_ratio * h + geo_map[feat_size + h * width + w];     // y1

                    pos_cur_out[count * 8 + 2] = pool_ratio * w + geo_map[2 * feat_size + h * width + w]; // x2
                    pos_cur_out[count * 8 + 3] = pool_ratio * h + geo_map[3 * feat_size + h * width + w]; // y2

                    pos_cur_out[count * 8 + 4] = pool_ratio * w + geo_map[4 * feat_size + h * width + w]; // x3
                    pos_cur_out[count * 8 + 5] = pool_ratio * h + geo_map[5 * feat_size + h * width + w]; // y3

                    pos_cur_out[count * 8 + 6] = pool_ratio * w + geo_map[6 * feat_size + h * width + w]; // x4
                    pos_cur_out[count * 8 + 7] = pool_ratio * h + geo_map[7 * feat_size + h * width + w]; // y4
                }
                else
                {
                    // RBOX mode
                    float up_pred    = geo_map[h * width + w];
                    float right_pred = geo_map[feat_size + h * width + w];
                    float down_pred  = geo_map[2 * feat_size + h * width + w];
                    float left_pred  = geo_map[3 * feat_size + h * width + w];
                    float angle_pred = geo_map[4 * feat_size + h * width + w];

                    pos_cur_out[count * 8]     = pool_ratio * w - up_pred *sin(angle_pred) - left_pred *cos(angle_pred);    // x1
                    pos_cur_out[count * 8 + 1] = pool_ratio * h - up_pred *cos(angle_pred) + left_pred *sin(angle_pred);    // y1

                    pos_cur_out[count * 8 + 2] = pool_ratio * w - up_pred *sin(angle_pred) + right_pred *cos(angle_pred);   // x2
                    pos_cur_out[count * 8 + 3] = pool_ratio * h - up_pred *cos(angle_pred) - right_pred *sin(angle_pred);   // y2

                    pos_cur_out[count * 8 + 4] = pool_ratio * w + down_pred *sin(angle_pred) + right_pred *cos(angle_pred); // x3
                    pos_cur_out[count * 8 + 5] = pool_ratio * h + down_pred *cos(angle_pred) - right_pred *sin(angle_pred); // y3

                    pos_cur_out[count * 8 + 6] = pool_ratio * w + down_pred *sin(angle_pred) - left_pred *cos(angle_pred);  // x4
                    pos_cur_out[count * 8 + 7] = pool_ratio * h + down_pred *cos(angle_pred) + left_pred *sin(angle_pred);  // y4
                }
                // Multiply data with as large number to prevent loss in nms
                for (int i = 0; i < 8; i++)
                {
                    pos_cur_out[count * 8 + i] *= EAST_OUT_PRECISION;
                }
                conf_cur_out[count] = score;
                count++;
            }
        }
    }

    // locality-aware NMS
    valid_box_num_all = EAST_OUT_lanms(pos_cur_out, conf_cur_out, count, nms_thresh);

    // standard NMS
    valid_box_num_all = EAST_OUT_nms(pos_cur_out, conf_cur_out, nms_thresh, valid_box_num_all);

    // Results formalized
    for (int i = 0; i < valid_box_num_all; i++)
    {
        float *p = &pos_data_out[i * EAST_OUT_POS_FORM];
        for (int j = 0; j < EAST_OUT_POS_FORM; j++)
        {
            result[i * EAST_OUT_OUTPUT_FORM + j] = p[j] / scale_factor;          // copy the predict boxes to output and rescale to original shapes
        }
        result[i * EAST_OUT_OUTPUT_FORM + EAST_OUT_POS_FORM] = conf_data_out[i]; // copy the confidence to output
    }

    *result_num = valid_box_num_all;

    // Free memory
    if (cache_data_copy != NULL)
    {
        free((void *)cache_data_copy);
        cache_data_copy = NULL;
    }
}
