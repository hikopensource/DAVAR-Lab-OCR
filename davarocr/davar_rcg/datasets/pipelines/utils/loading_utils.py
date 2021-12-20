"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    load_utils.py
# Abstract       :    Implementations of data loading utils

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import random
import math
import json
import os.path as osp

import cv2
import numpy as np


def wordmap_loader(wordmap, load_type):
    """

    Args:
        wordmap (str): file path of the recognition dictionary
        load_type (str): type of data loading, including ["LMDB", "LMDBOLD", "File", "Tight", "Loose"]

    Returns:
        str: content of the recognition dictionary
    Returns:
        set|str: support chars
    Returns:
        dict|None: character translate table
    """

    # character dictionary is file format
    if osp.isfile(wordmap):
        print("--------------------------------------------------------------------------------")
        print(" -- RCGLoadImageFrom %s loading characters from user predefined file !!! --" % load_type)
        print("--------------------------------------------------------------------------------")

        # character dictionary is json format
        if wordmap.endswith('.json'):
            with open(wordmap, 'r', encoding='utf-8') as character_file:
                character = json.load(character_file)
                assert 'char2index' in character
                if 're_chars' in character:
                    character_tmp = character['re_chars']
                else:
                    character_tmp = ''.join(list(character['char2index'].keys()))
                support_chars = set(character['char2index'].keys())
                if 'char2char' in character:
                    table = dict()
                    for key, value in character['char2char'].items():
                        table[ord(key)] = ord(value)
            return character_tmp, support_chars, table
        # character dictionary is txt format
        elif wordmap.endswith('.txt'):
            with open(wordmap, 'r', encoding='utf-8') as character_file:
                character = character_file.readline().strip()
                character_tmp = character
                if len(character) > 1000:
                    character_tmp = character.replace(u'¡ó', '').\
                        replace('\\', '\\\\').\
                        replace('[', '\[').\
                        replace(']', '\]').\
                        replace('-', '\-')
            return character_tmp, character_tmp, None
        else:
            raise Exception("wordmap file type is not support !!!")
    elif ".json" in wordmap or ".txt" in wordmap:
        raise FileNotFoundError("The recognition wordmap file is not existing")
    # character dictionary is character
    else:
        print("--------------------------------------------------------------------------------")
        print(" -- RCGLoadImageFrom %s loading characters from user predefined word !!! --" % load_type)
        print("--------------------------------------------------------------------------------")
        character = wordmap
        return character, character, None


def clc_points(points):
    """

    Args:
        points (np.array|list): OpenCV cv2.boxPoints returns coordinates, order is
                                [right_bottom, left_bottom, left_top, right_top]

    Returns:
        list: reorder the coordinates, order is [left_top, right_top, right_bottom, left_bottom]

    """
    pt_list = list(map(lambda x: list(map(int, x)), list(points)))
    temp_box = sorted(pt_list, key=lambda x: x[0])

    left_pt = temp_box[:2]
    right_pt = temp_box[2:]

    # sort the coordinate
    left_pt = sorted(left_pt, key=lambda x: x[1])
    right_pt = sorted(right_pt, key=lambda x: x[1])
    res_list = [left_pt[0], right_pt[0], right_pt[1], left_pt[1]]
    return np.float32(res_list)


def crop_and_transform(img, bbox, crop_only=False):
    """
    Args:
        img (np.array): input image
        bbox (np.array): the coordinate of the crop image
        crop_only (bool): whether to crop image with the perspective transformation

    Returns:
        np.array: Cropped and transformed image
    """

    points = np.int32(bbox).reshape(4, 2)

    min_x, max_x, min_y, max_y = min(points[:, 0]), max(points[:, 0]), min(points[:, 1]), max(points[:, 1])

    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(img.shape[1], max_x), min(img.shape[0], max_y)

    if len(img.shape) == 2:
        crop_img = img[min_y: max_y, min_x: max_x]
    else:
        crop_img = img[min_y: max_y, min_x: max_x, :]

    if crop_only:
        return crop_img

    points[:, 0] -= min_x
    points[:, 1] -= min_y

    rec_points = cv2.boxPoints(cv2.minAreaRect(points))

    c_points = clc_points(rec_points)

    width = int(np.linalg.norm(c_points[1] - c_points[0]))
    height = int(np.linalg.norm(c_points[2] - c_points[1]))
    dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # perspective transformation
    rotate_mat = cv2.getPerspectiveTransform(c_points, dst_points)
    dst = cv2.warpPerspective(crop_img, rotate_mat, (width, height))
    return dst


def shake_crop(img, bbox, crop_pixel_shake=None, need_crop=True):
    """

    Args:
        img (np.array): input image
        bbox (np.array): the coordinate of the crop image
        crop_pixel_shake (list): parameter of the pixel shake
        need_crop (bool): whether to crop the image

    Returns:
        np.array: Cropped and pixel shaked image

    """

    points = np.int32(bbox).reshape(4, 2)
    min_x, max_x, min_y, max_y = min(points[:, 0]), max(points[:, 0]), min(points[:, 1]), max(points[:, 1])

    if not need_crop:
        bbox_new = [0, 0, img.shape[1], 0, img.shape[1], img.shape[0], 0, img.shape[0]]
        points = np.int32(bbox_new).reshape(4, 2)
        min_x, max_x, min_y, max_y = min(points[:, 0]), max(points[:, 0]), min(points[:, 1]), max(points[:, 1])

    if crop_pixel_shake is not None and need_crop:
        # int: shake pixels
        if isinstance(crop_pixel_shake[0], int):
            inside_x, outside_x, inside_y, outside_y = crop_pixel_shake

        # float: shake percent
        elif isinstance(crop_pixel_shake[0], float):
            d_y = max_y - min_y
            inside_x = int(d_y * crop_pixel_shake[0])
            outside_x = int(d_y * crop_pixel_shake[1])
            inside_y = int(d_y * crop_pixel_shake[2])
            outside_y = int(d_y * crop_pixel_shake[3])
        else:
            raise TypeError('Unsupport crop_pixel_shake type:', crop_pixel_shake)

        # random pixel shake
        min_x += random.randint(-outside_x, inside_x)
        min_y += random.randint(-outside_y, inside_y)
        max_x += random.randint(-inside_x, outside_x)
        max_y += random.randint(-inside_y, outside_y)

    min_x, min_y = max(0, min_x), max(0, min_y)
    max_x, max_y = min(img.shape[1], max_x), min(img.shape[0], max_y)

    if len(img.shape) == 2:
        crop_img = img[min_y: max_y, min_x: max_x]
    else:
        crop_img = img[min_y: max_y, min_x: max_x, :]
    return crop_img


def check_point(point, img_w, img_h):
    """

    Args:
        point (np.array): coordinates
        img_w (int): image width
        img_h (int): image height

    Returns:
        bool: legality of the coordinates

    """
    return 0 <= point[0] <= img_w and 0 <= point[1] <= img_h


def shake_point(img, bbox, crop_pixel_shake=None):
    """

    Args:
        img (np.array): input image
        bbox (np.array): bounding box
        crop_pixel_shake (list): parameter of the pixel shake

    Returns:
        list: bounding box after the shaking pixel

    """
    if crop_pixel_shake is None:
        return bbox

    ratio_obd_w = random.random() * crop_pixel_shake[1]
    ratio_obd_h = random.random() * crop_pixel_shake[3]
    contour = np.array(bbox).reshape(-1, 2)
    contour_tmp = np.zeros((4, 2), dtype=np.int)

    img_h = img.shape[0]
    img_w = img.shape[1]

    x_delta = contour[1][0] - contour[0][0]
    y_delta = contour[1][1] - contour[0][1]
    contour_tmp[0][0] = contour[0][0] - int(x_delta * ratio_obd_w)
    contour_tmp[0][1] = contour[0][1] - int(y_delta * ratio_obd_w)
    contour_tmp[1][0] = contour[1][0] + int(x_delta * ratio_obd_w)
    contour_tmp[1][1] = contour[1][1] + int(y_delta * ratio_obd_w)

    x_delta = contour[2][0] - contour[3][0]
    y_delta = contour[2][1] - contour[3][1]
    contour_tmp[3][0] = contour[3][0] - int(x_delta * ratio_obd_w)
    contour_tmp[3][1] = contour[3][1] - int(y_delta * ratio_obd_w)
    contour_tmp[2][0] = contour[2][0] + int(x_delta * ratio_obd_w)
    contour_tmp[2][1] = contour[2][1] + int(y_delta * ratio_obd_w)

    if check_point(contour_tmp[0], img_w, img_h) and check_point(contour_tmp[3], img_w, img_h):
        contour[0][0] = min(max(contour_tmp[0][0], 0), img_w - 1)
        contour[0][1] = min(max(contour_tmp[0][1], 0), img_h - 1)
        contour[3][0] = min(max(contour_tmp[3][0], 0), img_w - 1)
        contour[3][1] = min(max(contour_tmp[3][1], 0), img_h - 1)

    if check_point(contour_tmp[1], img_w, img_h) and check_point(contour_tmp[2], img_w, img_h):
        contour[1][0] = min(max(contour_tmp[1][0], 0), img_w - 1)
        contour[1][1] = min(max(contour_tmp[1][1], 0), img_h - 1)
        contour[2][0] = min(max(contour_tmp[2][0], 0), img_w - 1)
        contour[2][1] = min(max(contour_tmp[2][1], 0), img_h - 1)

    contour_tmp = np.zeros((4, 2), dtype=np.int)

    x_delta = contour[3][0] - contour[0][0]
    y_delta = contour[3][1] - contour[0][1]
    contour_tmp[0][0] = contour[0][0] - int(x_delta * ratio_obd_h)
    contour_tmp[0][1] = contour[0][1] - int(y_delta * ratio_obd_h)
    contour_tmp[3][0] = contour[3][0] + int(x_delta * ratio_obd_h)
    contour_tmp[3][1] = contour[3][1] + int(y_delta * ratio_obd_h)

    x_delta = contour[2][0] - contour[1][0]
    y_delta = contour[2][1] - contour[1][1]
    contour_tmp[1][0] = contour[1][0] - int(x_delta * ratio_obd_h)
    contour_tmp[1][1] = contour[1][1] - int(y_delta * ratio_obd_h)
    contour_tmp[2][0] = contour[2][0] + int(x_delta * ratio_obd_h)
    contour_tmp[2][1] = contour[2][1] + int(y_delta * ratio_obd_h)

    if check_point(contour_tmp[0], img_w, img_h) and check_point(contour_tmp[1], img_w, img_h):
        contour[0][0] = min(max(contour_tmp[0][0], 0), img_w - 1)
        contour[0][1] = min(max(contour_tmp[0][1], 0), img_h - 1)
        contour[1][0] = min(max(contour_tmp[1][0], 0), img_w - 1)
        contour[1][1] = min(max(contour_tmp[1][1], 0), img_h - 1)

    if check_point(contour_tmp[3], img_w, img_h) and check_point(contour_tmp[2], img_w, img_h):
        contour[3][0] = min(max(contour_tmp[3][0], 0), img_w - 1)
        contour[3][1] = min(max(contour_tmp[3][1], 0), img_h - 1)
        contour[2][0] = min(max(contour_tmp[2][0], 0), img_w - 1)
        contour[2][1] = min(max(contour_tmp[2][1], 0), img_h - 1)

    contour = list(map(int, contour.reshape(-1)))
    return contour


def scale_box(pos_cor, src_h, src_w, ratio=1.1):
    """

    Args:
        pos_cor (np.array): bounding box
        src_h (int): image height
        src_w (int): image width
        ratio (float): ratios of the fixed expand

    Returns:
        np.array: bounding box after the expanding

    """
    ratio_w = ratio - 1

    # expand the bounding box at left direction and right direction
    x_delta = pos_cor[2] - pos_cor[0]
    y_delta = pos_cor[3] - pos_cor[1]

    pos_cor[0] = pos_cor[0] - int(x_delta * ratio_w / 2)
    pos_cor[1] = pos_cor[1] - int(y_delta * ratio_w / 2)
    pos_cor[2] = pos_cor[2] + int(x_delta * ratio_w / 2)
    pos_cor[3] = pos_cor[3] + int(y_delta * ratio_w / 2)

    # expand the bounding box at top direction and bottom direction
    x_delta = pos_cor[4] - pos_cor[6]
    y_delta = pos_cor[5] - pos_cor[7]
    pos_cor[4] = pos_cor[4] + int(x_delta * ratio_w / 2)
    pos_cor[5] = pos_cor[5] + int(y_delta * ratio_w / 2)
    pos_cor[6] = pos_cor[6] - int(x_delta * ratio_w / 2)
    pos_cor[7] = pos_cor[7] - int(y_delta * ratio_w / 2)

    # constraint on the bounding box coordinate legal
    for i in range(0, 8, 2):
        pos_cor[i] = min(max(0, pos_cor[i]), src_w - 1)
        pos_cor[i + 1] = min(max(0, pos_cor[i + 1]), src_h - 1)
    return pos_cor


def scale_box_hori_vert(pos_cor, src_h, src_w, ratio):
    """
        calculate circumscribed rectangle to bounding box, expand it horizontally and vertically

    Args:
        pos_cor (np.array): bounding box coordinate
        src_h (int): image height
        src_w (int): image width
        ratio (list): ratios of the expanding,
                        1. ratio[0] means the ratio of horizontally expanding,
                        2. ratio[1] means the ratio of vertically expanding,

    Returns:
        list: bounding box after horizontally and vertically expanding

    """
    assert len(ratio) == 2
    tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y = pos_cor
    x_min = min(tl_x, tr_x, br_x, bl_x)
    x_max = max(tl_x, tr_x, br_x, bl_x)
    y_min = min(tl_y, tr_y, br_y, bl_y)
    y_max = max(tl_y, tr_y, br_y, bl_y)

    # calculate the height and width of the bounding box
    width = x_max - x_min
    height = y_max - y_min

    hori_percent, vert_percent = ratio

    # expand the bounding box horizontally and vertically
    x_min -= (width * hori_percent)
    x_max += (width * hori_percent)
    y_min -= (height * vert_percent)
    y_max += (height * vert_percent)

    x_min = max(0, x_min)
    x_max = min(src_w, x_max)
    y_min = max(0, y_min)
    y_max = min(src_h, y_max)

    pos_cor = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

    return pos_cor


def scale_point_hori_vert(contour, img_h, img_w, ratio):
    """
        expand the bounding box horizontally and vertically

    Args:
        contour (list): bounding box coordinate
        img_h (int): image height
        img_w (int): image width
        ratio (list): ratios of the expanding,
                        1. ratio[0] means the ratio of horizontally expanding,
                        2. ratio[1] means the ratio of vertically expanding,
    Returns:
        list: bounding box after horizontally and vertically expanding
    """
    assert len(ratio) == 2
    ratio_obd_w, ratio_obd_h = ratio
    contour = np.array(contour).reshape(-1, 2)
    contour_tmp = np.zeros((4, 2), dtype=np.int)

    x_delt = contour[1][0] - contour[0][0]
    y_delt = contour[1][1] - contour[0][1]

    contour_tmp[0][0] = contour[0][0] - int(x_delt * ratio_obd_w)
    contour_tmp[0][1] = contour[0][1] - int(y_delt * ratio_obd_w)
    contour_tmp[1][0] = contour[1][0] + int(x_delt * ratio_obd_w)
    contour_tmp[1][1] = contour[1][1] + int(y_delt * ratio_obd_w)

    x_delt = contour[2][0] - contour[3][0]
    y_delt = contour[2][1] - contour[3][1]

    contour_tmp[3][0] = contour[3][0] - int(x_delt * ratio_obd_w)
    contour_tmp[3][1] = contour[3][1] - int(y_delt * ratio_obd_w)
    contour_tmp[2][0] = contour[2][0] + int(x_delt * ratio_obd_w)
    contour_tmp[2][1] = contour[2][1] + int(y_delt * ratio_obd_w)

    if check_point(contour_tmp[0], img_w, img_h) and check_point(contour_tmp[3], img_w, img_h):
        contour[0][0] = min(max(contour_tmp[0][0], 0), img_w - 1)
        contour[0][1] = min(max(contour_tmp[0][1], 0), img_h - 1)
        contour[3][0] = min(max(contour_tmp[3][0], 0), img_w - 1)
        contour[3][1] = min(max(contour_tmp[3][1], 0), img_h - 1)

    if check_point(contour_tmp[1], img_w, img_h) and check_point(contour_tmp[2], img_w, img_h):
        contour[1][0] = min(max(contour_tmp[1][0], 0), img_w - 1)
        contour[1][1] = min(max(contour_tmp[1][1], 0), img_h - 1)
        contour[2][0] = min(max(contour_tmp[2][0], 0), img_w - 1)
        contour[2][1] = min(max(contour_tmp[2][1], 0), img_h - 1)

    contour_tmp = np.zeros((4, 2), dtype=np.int)

    x_delt = contour[3][0] - contour[0][0]
    y_delt = contour[3][1] - contour[0][1]

    contour_tmp[0][0] = contour[0][0] - int(x_delt * ratio_obd_h)
    contour_tmp[0][1] = contour[0][1] - int(y_delt * ratio_obd_h)
    contour_tmp[3][0] = contour[3][0] + int(x_delt * ratio_obd_h)
    contour_tmp[3][1] = contour[3][1] + int(y_delt * ratio_obd_h)

    x_delt = contour[2][0] - contour[1][0]
    y_delt = contour[2][1] - contour[1][1]

    contour_tmp[1][0] = contour[1][0] - int(x_delt * ratio_obd_h)
    contour_tmp[1][1] = contour[1][1] - int(y_delt * ratio_obd_h)
    contour_tmp[2][0] = contour[2][0] + int(x_delt * ratio_obd_h)
    contour_tmp[2][1] = contour[2][1] + int(y_delt * ratio_obd_h)

    if check_point(contour_tmp[0], img_w, img_h) and check_point(contour_tmp[1], img_w, img_h):
        contour[0][0] = min(max(contour_tmp[0][0], 0), img_w - 1)
        contour[0][1] = min(max(contour_tmp[0][1], 0), img_h - 1)
        contour[1][0] = min(max(contour_tmp[1][0], 0), img_w - 1)
        contour[1][1] = min(max(contour_tmp[1][1], 0), img_h - 1)

    if check_point(contour_tmp[3], img_w, img_h) and check_point(contour_tmp[2], img_w, img_h):
        contour[3][0] = min(max(contour_tmp[3][0], 0), img_w - 1)
        contour[3][1] = min(max(contour_tmp[3][1], 0), img_h - 1)
        contour[2][0] = min(max(contour_tmp[2][0], 0), img_w - 1)
        contour[2][1] = min(max(contour_tmp[2][1], 0), img_h - 1)

    contour = list(map(int, contour.reshape(-1)))
    return contour


def get_two_point_dis(point1, point2):
    """

    Args:
        point1 (list): point1
        point2 (list): point2

    Returns:
        int: Euclidean distance between point1 and point2

    """

    # calculate the euclidean distance
    dist = math.sqrt((point1[0] - point2[0]) *
                     (point1[0] - point2[0])
                     + (point1[1] - point2[1]) *
                     (point1[1] - point2[1]))
    return int(dist)


def get_rectangle_img(src_img, pos_cor):
    """

    Args:
        src_img (np.array): source image
        pos_cor (list): crop coordinate

    Returns:
        np.array: cropped image
    Returns:
        np.array: coordinate of the cropped image
    Returns:
        bool: comparison on the angle with the horizontal direction is less than 15

    """

    cur_width = get_two_point_dis([pos_cor[0], pos_cor[1]], [pos_cor[2], pos_cor[3]])
    cur_height = get_two_point_dis([pos_cor[0], pos_cor[1]], [pos_cor[6], pos_cor[7]])

    height = src_img.shape[0]
    width = src_img.shape[1]

    points = np.int32(pos_cor).reshape(4, 2)
    minx = max(0, min(points[:, 0]))
    maxx = min(width-1, max(points[:, 0]))
    miny = max(0, min(points[:, 1]))
    maxy = min(height-1, max(points[:, 1]))

    points[:, 0] -= minx
    points[:, 1] -= miny
    pos_cor = points.flatten().tolist()

    vector = points[1, :] - points[0, :]  # point 0->1, construct the vector
    horizon_vector = np.int32([1, 0])     # horizontal direction vector
    l_v = np.sqrt(vector.dot(vector))
    l_h = np.sqrt(horizon_vector.dot(horizon_vector))
    cos_angle = vector.dot(horizon_vector) / (l_v * l_h + 0.0000001)
    angle = np.arccos(cos_angle)  # calculate the angle, 0~pi [arccos(1)=0, arccos(-1)=pi]

    cn_width = int((maxx - minx + 2) // 2 * 2)
    cn_height = int((maxy - miny + 2) // 2 * 2)

    is_satisfy = False
    # horizontal angle is less than 15°(0.26 Radian system)
    if (cur_width * cur_height) > 0.8 * (cn_width * cn_height + 0.0000001) and angle < 0.26:
        is_satisfy = True
    if len(src_img.shape) == 3:
        return src_img[miny:maxy + 1, minx:maxx + 1, :], pos_cor, is_satisfy
    return src_img[miny:maxy + 1, minx:maxx + 1], pos_cor, is_satisfy


def get_perspective_img(src_img, pos_cor):
    """

    Args:
        src_img (np.array): input image
        pos_cor (list): crop image coordinate

    Returns:
        np.array: image after the perspective transformation

    """

    # Calculate the horizontal bounding rectangle
    rectangle_img, pos_cor, is_satisfy = get_rectangle_img(src_img, pos_cor)

    # use the horizontal bounding rectangle and resize
    if is_satisfy:
        dst_img = rectangle_img
    # crop image with the perspective transformation
    else:
        cur_width = get_two_point_dis([pos_cor[0], pos_cor[1]], [pos_cor[2], pos_cor[3]])
        cur_height = get_two_point_dis([pos_cor[0], pos_cor[1]], [pos_cor[6], pos_cor[7]])
        dst_point = np.array([(0, 0), (cur_width-1, 0), (cur_width-1, cur_height-1), (0, cur_height-1)],
                             dtype=np.float32)
        src_point = np.float32(pos_cor).reshape((4, 2))
        trans_mat = cv2.getPerspectiveTransform(src_point, dst_point)
        dst_img = cv2.warpPerspective(rectangle_img, trans_mat, (cur_width, cur_height))

    return dst_img


def trans_rot_affine(matrix, u_vec, v_vec):
    """

    Args:
        matrix (matrix): rotation matrix
        u_vec (float): x coordinate
        v_vec (float): y coordinate

    Returns:
        int: rotated x coordinate
    Returns:
        int: rotated y coordinate

    """

    # rotation affine transformation
    x_vec = u_vec * matrix[0][0] + v_vec * matrix[0][1] + matrix[0][2]
    y_vec = u_vec * matrix[1][0] + v_vec * matrix[1][1] + matrix[1][2]
    return int(x_vec), int(y_vec)


def trans_rot_text_box(matrix, box1):
    """

    Args:
        matrix (matrix): rotate matrix
        box1 (list): bounding box coordinate

    Returns:
        list: bounding box coordinate after rotation

    """

    # rotation affine transformation on text image

    box2 = [0]*8
    box2[0], box2[1] = trans_rot_affine(matrix, box1[0], box1[1])
    box2[2], box2[3] = trans_rot_affine(matrix, box1[2], box1[3])
    box2[4], box2[5] = trans_rot_affine(matrix, box1[4], box1[5])
    box2[6], box2[7] = trans_rot_affine(matrix, box1[6], box1[7])
    return box2


def get_poly_angle(pos_cor):
    """
        calculate the rotation angle

    Args:
        pos_cor (list): bounding box

    Returns:
        float: rotation angle

    """
    pi_num = 3.141592653
    pos_cor = list(map(float, pos_cor))

    x_1 = pos_cor[0]
    y_1 = pos_cor[1]
    x_2 = pos_cor[2]
    y_2 = pos_cor[3]

    x_x = x_2 - x_1
    y_y = y_2 - y_1

    if x_x == 0.0:
        angle_temp = pi_num / 2.0
    else:
        angle_temp = math.atan(abs(y_y / x_x))

    if (x_x < 0.0) and (y_y >= 0.0):
        angle_temp = pi_num - angle_temp
    elif (x_x < 0.0) and (y_y < 0.0):
        angle_temp = pi_num + angle_temp
    elif (x_x >= 0.0) and (y_y < 0.0):
        angle_temp = pi_num * 2.0 - angle_temp
    else:
        angle_temp = math.atan(abs(y_y / x_x))

    return angle_temp / pi_num * 180


def poly_to_rect(pos_cor, height, width):
    """
        Polygon to rectangle
    Args:
        pos_cor (list): x1, y1, x2, y2, x3, y3, x4, y4
        height (int): image height
        width (int): image width

    Returns:
        list: rect: x, y, width, height

    """
    x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = pos_cor

    # calculate the circumscribed rectangle coordinate
    dst_x1 = max(0, min(x_1, x_2, x_3, x_4))
    dst_y1 = max(0, min(y_1, y_2, y_3, y_4))
    dst_x2 = min(width - 1, max(x_1, x_2, x_3, x_4))
    dst_y2 = min(height - 1, max(y_1, y_2, y_3, y_4))

    dst_x, dst_y = dst_x1, dst_y1
    width = dst_x2 - dst_x1 + 1
    height = dst_y2 - dst_y1 + 1

    return [dst_x, dst_y, width, height]


def rotate_and_crop(cv_image, old_pos_cor, max_angle=0,
                    random_crop=True, random_width=0.2,
                    random_height=0.2, **kwargs):
    """
    online crop image, rotate the text horizontally and crop image
    Args:
        cv_image (np.array): image
        old_pos_cor (list): bounding box coordinate
        max_angle (int): max rotate angle
        random_crop (bool): whether to random crop, training stage sets True
        random_width (int): random width, expand roi_width range (1+[0, random_width])*roi_width
        random_height (int): random height, expand roi_height range(1+[0, random_height])*roi_height
        **kwargs (None): backup parameter

    Returns:
        np.array: image after rotation and crop

    """
    random_th = 0.3
    pos_cor = old_pos_cor.copy()

    # utilize the first two points to calculate the angle and crop image
    rotate_angle = get_poly_angle(pos_cor)

    if random.random() > random_th:
        random_angle = (random.random() * 2 - 1) * max_angle  # [-1, 1] * max_angle
        rotate_angle += random_angle

    height, width = cv_image.shape[:2]
    roi_x, roi_y, roi_width, roi_height = poly_to_rect(pos_cor, height, width)

    # get a larger roi rect, because random crop is operated in the next
    roi_larger_x = int(max(0, roi_x - roi_width / 2))
    roi_larger_y = int(max(0, roi_y - roi_height))
    right_x = int(min(width, roi_x + roi_width / 2 * 3))
    bottom_y = int(min(height, roi_y + 2 * roi_height))
    roi_larger_img = cv_image[roi_larger_y:bottom_y, roi_larger_x:right_x, :]

    # get the new coord in roi_larger_img
    for label_i in range(4):
        pos_cor[2 * label_i] -= roi_larger_x
        pos_cor[2 * label_i + 1] -= roi_larger_y

    roi_larger_h, roi_larger_w = roi_larger_img.shape[:2]
    radis = int(math.sqrt(roi_larger_h ** 2 + roi_larger_w ** 2)) + 2
    larger_img_for_rotate = np.zeros((radis, radis, 3))
    roi_x = int((radis - roi_larger_w) / 2)
    roi_y = int((radis - roi_larger_h) / 2)
    larger_img_for_rotate[roi_y:roi_y+roi_larger_h, roi_x:roi_x+roi_larger_w, :] = roi_larger_img

    # get the new coord in larger_img_for_rotate
    for label_i in range(4):
        pos_cor[2 * label_i] += roi_x
        pos_cor[2 * label_i + 1] += roi_y
    height, width = larger_img_for_rotate.shape[:2]

    rotate_matrix = cv2.getRotationMatrix2D((height / 2, width / 2), rotate_angle, 1)

    rotate_img = cv2.warpAffine(larger_img_for_rotate, rotate_matrix, (height, width))

    pos_cor = trans_rot_text_box(rotate_matrix, pos_cor)
    height, width = rotate_img.shape[:2]
    roi_x, roi_y, roi_width, roi_height = poly_to_rect(pos_cor, height, width)

    # random crop
    if random_crop and (random.random() > random_th):
        add_width = int(random_width * roi_width * random.random())
        add_height = int(random_height * roi_height * random.random())
        per_w = random.random()
        per_h = random.random()
        roi_x = max(0, roi_x - int(per_w * add_width))
        roi_y = max(0, roi_y - int(per_h * add_height))
        x_rt = min(width - 1, (roi_x + add_width + roi_width))
        y_rt = min(height - 1, (roi_y + add_height + roi_height))
        roi_width = max(1, (x_rt - roi_x))
        roi_height = max(1, (y_rt - roi_y))

    # text image after rotation
    roi_img = rotate_img[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width, :]
    return roi_img
