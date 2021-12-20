"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    util_poly.py
# Abstract       :    This file contains some functions for processing polygons

# Current Version:    1.0.0
# Date           :    2021-09-01
##################################################################################################
"""
import math
import numpy as np


def point_distance(point_a, point_b):
    """ Calculate distance between to points.

    Args:
        point_a (list(float)): coordinate of the first point, [x, y]
        point_b (list(float)): coordinate of the second point, [x, y]

    Returns:
        float: distance between the two points.
    """
    return math.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)

def dot_product(point_a, point_b, point_c, point_d):
    """ Calculate dot product for two vectors.

    Args:
        point_a (list(float)): start point of vector 1.
        point_b (list(float)): end point of vector 1.
        point_c (list(float)): start point of vector 2.
        point_d (list(float)): end point of vector 2.

    Returns:
        float: dot product of vector 1 and vector 2
    """
    return (point_b[0] - point_a[0]) * (point_d[1] - point_c[1]) - \
    (point_d[0] - point_c[0]) * (point_b[1] - point_a[1])

def get_angle(point_a, point_b, point_c):
    """ Calculate angle of point_a->point_c->point_b

    Args:
        point_a (list(float)): start point
        point_b (list(float)): middle point
        point_c (list(float)): end point

    Returns:
        float: angle degree in [0, 360]
    """
    theta = math.atan2(point_a[0] - point_c[0], point_a[1] - point_c[1]) - math.atan2(point_b[0] - point_c[0],
                                                                                      point_b[1] - point_c[1])
    if theta > math.pi:
        theta -= 2 * math.pi
    if theta < - math.pi:
        theta += 2 * math.pi
    theta = theta * 180.0 / math.pi
    if theta < 0:
        theta = - theta
    if dot_product(point_a, point_c, point_c, point_b) < 0:
        theta = 360 - theta
    return theta

def get_quad_index(polys):
    """ Estimate the corner points indexes and to make the top-left point as the index 0.
        For vertical instances, make the right-left point as the index 0.
        e.g., for quadrangle, the order is top-lef, top-right, bottom-right, bottom-left, respectively.

    Args:
        polys (list(list(float)): points of the polygon boxes. [ [x1, y1], ..., [xn, yn]]

    Returns:
        list(int): re-ordered corner points indexes.
    """
    # 4-point quadrangle shapes
    if polys.shape[0] == 4:
        tmp = np.zeros(4)
        tmp[0] = 0
        tmp[1] = 1
        tmp[2] = 2
        tmp[3] = 3

        # For vertical instances, make top-right point as the first point.
        if point_distance(polys[0], polys[3]) > 2 * point_distance(polys[0], polys[1]):
            tmp[0] = 1
            tmp[1] = 2
            tmp[2] = 3
            tmp[3] = 0
        return tmp
    angles = np.zeros(polys.shape[0])

    # The neighbor boundaries of head and tail boundary are nearly parrallel
    for i in range(polys.shape[0]):
        angle1 = get_angle(polys[i - 1], polys[(i + 1) % polys.shape[0]], polys[i])
        angle2 = get_angle(polys[i - 2], polys[i], polys[i - 1])
        angles[i] = abs(angle1 + angle2 - 180.0)
    tmp_index = 1
    ret = np.zeros(4)
    index = np.argsort(angles)
    while abs(index[0] - index[tmp_index]) == 1 or abs(index[0] - index[tmp_index]) == polys.shape[0] - 1:
        tmp_index += 1
        if tmp_index == len(index):
            return ret
    if index[0] < index[tmp_index]:
        ret[0] = index[0]
        ret[1] = (index[tmp_index] - 1 + polys.shape[0]) % polys.shape[0]
        ret[2] = index[tmp_index]
        ret[3] = (index[0] - 1 + polys.shape[0]) % polys.shape[0]
    else:
        ret[0] = index[tmp_index]
        ret[1] = (index[0] - 1 + polys.shape[0]) % polys.shape[0]
        ret[2] = index[0]
        ret[3] = (index[tmp_index] - 1 + polys.shape[0]) % polys.shape[0]
    return ret

def get_sample_point(polys, sample_point_number):
    """ Averagely sample key points on top & bottom boundary of the polygon contour.

    Args:
        polys (list(list(int)): polygon bounding boxes of text instances.
        sample_point_number (int): point numbers in top / bottom boundary, which will generate 2*M points.

    Returns:
        list(list(int)): sampled points, in shape of [N, 2*M]
    """
    distance = np.zeros(polys.shape[0])
    length = np.zeros(polys.shape[0])
    sample_point = np.zeros((sample_point_number, 2))

    # Calculate the distance between adjacent points
    for i in range(polys.shape[0]-1):
        distance[i+1] = point_distance(polys[i], polys[i+1])

    # Calculate the distance between the 0-th point and the i-th point
    for i in range(1, polys.shape[0]):
        length[i] = length[i-1] + distance[i]

    # Calculate the average distance between adjacent points
    avg_distance = np.sum(distance) / (sample_point_number - 1)

    # Averagely sample points along the polygon
    for i in range(sample_point_number-1):
        cur_pos = avg_distance * i
        for j in range(polys.shape[0]-1):
            if length[j] <= cur_pos < length[j+1]:
                sample_point[i] = (polys[j + 1] - polys[j]) * \
                (cur_pos - length[j]) / (length[j + 1] - length[j]) + polys[j]
    sample_point[-1] = polys[-1]
    return sample_point

