# -*- coding: utf-8 -*-
import os
import argparse

import numpy as np
import Polygon as plg
from scipy.optimize import linear_sum_assignment
import Levenshtein

string_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz[]+-#$()@=_!?,:;/.%&\\\'\">*|<`{~}^   "


def parse_args():
    """

    Returns:
        args parameter of model test

    """
    parser = argparse.ArgumentParser(description='DavarOCR video text end-to-end evaluation')
    parser.add_argument('predict_file', help='predict results')
    parser.add_argument('gt_file', help='gt file')
    parser.add_argument('--voca_file', type=str, default=None)
    parser.add_argument('--care_rcg', type=int, default=1)

    args_ = parser.parse_args()
    return args_


def polygon_from_points(points):
    """make a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4

    Args:
        points (list): coordinate for box with 8 points:x1,y1,x2,y2,x3,y3,x4,y4
    Returns:
        Polygon: Polygon object

    """
    res_boxes = np.empty([1, 8], dtype='int32')
    res_boxes[0, 0] = int(points[0])
    res_boxes[0, 4] = int(points[1])
    res_boxes[0, 1] = int(points[2])
    res_boxes[0, 5] = int(points[3])
    res_boxes[0, 2] = int(points[4])
    res_boxes[0, 6] = int(points[5])
    res_boxes[0, 3] = int(points[6])
    res_boxes[0, 7] = int(points[7])
    point_mat = res_boxes[0].reshape([2, 4]).T
    return plg.Polygon(point_mat)


def get_union(p_d, p_g):
    """Get two polygons' union area

    Args:
        p_d (Polygon): Polygon object
        p_g (Polygon): Polygon object

    Returns:
        float: the union area between pD and pG

    """
    area_a = p_d.area()
    area_b = p_g.area()
    return area_a + area_b - get_intersection(p_d, p_g)


def get_intersection(p_d, p_g):
    """Get two polygons' intersection area

    Args:
        p_d (Polygon): Polygon object
        p_g (Polygon): Polygon object

    Returns:
        float: the intersection area between pD and pG

    """
    p_inter = p_d & p_g
    if len(p_inter) == 0:
        return 0
    return p_inter.area()


def get_intersection_over_union(p_d, p_g):
    """Get two polygons' IOU

    Args:
        p_d (Polygon): Polygon object
        p_g (Polygon): Polygon object

    Returns:
        float: the IOU area between p_d and p_g

    """
    try:
        return get_intersection(p_d, p_g) / get_union(p_d, p_g)
    except ZeroDivisionError:
        return 0


def Hungary(task_matrix):
    """Use Hungary algorithm to calculate the maximum match matrix

    Args:
        task_matrix (numpy array): two-dimensional matrix

    Returns:
       list(int): the row indices of task_matrix

    Returns:
        list(int): the matched col indices of task_matrix

    """
    row_ind, col_ind = linear_sum_assignment(task_matrix, maximize=True)
    return row_ind, col_ind


def process_gt_trans(trans, voca_list):
    """Correct gt trans

    Args:
        trans (str): original gt trans
        voca_list(list(str)): vocabulary list

    Returns:
       str: corrected trans


    """
    contain_number = False
    for i in range(len(trans)):
        cur_index = string_map.index(trans[i])
        if 0 <= cur_index <= 9:
            contain_number = True

    min_gt_dist = 1e7
    gt_word = ''
    if not trans == '###' and not contain_number:
        for voca in voca_list:
            cur_dist = Levenshtein.distance(voca, trans)
            if cur_dist < min_gt_dist:
                min_gt_dist = cur_dist
                gt_word = voca

    # For number or "###": using original trans
    else:
        gt_word = trans
    return gt_word


def load_gt_ata(gt_dict, voca_list):
    """Loading gt data for ata

    Args:
        gt_dict (dict): gt data
        voca_list(list(str)): vocabulary list
        
    Returns:
       dict(): save  all frame's bbox for all track sequences in a video

    Returns:
       dict(): save all frame's bbox quality for all track sequences in a video

    Returns:
       dict(): save all track sequences' recognition result in a video

    Returns:
       dict(): the number of not care gt track sequence

    """
    # To save  all frame's bbox for all track sequences in a video
    gt_seq_dict = dict()
    # To save all frame's bbox quality for all track sequences in a video e.g: "LOW", "MODERATE", "HIGH"
    gt_seq_quality_dict = dict()
    # To save all track sequences' recognition result in a video
    gt_seq_trans_dict = dict()
    # The number of not care gt track sequence
    gt_notcare_num = 0

    for gt_seq_ind in gt_dict:
        trans = gt_dict[gt_seq_ind]['trans']

        if voca_list:
            # Correct the gt trans for official IC15
            trans = process_gt_trans(trans, voca_list)

        if trans == '###':
            gt_notcare_num = gt_notcare_num + 1

        seq_frame_loc_dict = dict()
        seq_frame_quality_dict = dict()

        for instance in gt_dict[gt_seq_ind]['track']:
            frame_id = int(instance.split(',', 2)[0])
            point_list = []
            seq_frame_quality_dict[frame_id] = instance.rsplit(',', 2)[-2]
            bbox = instance.rsplit(',', 2)[-1]
            for i in range(8):
                point_list.append(np.int(bbox.split('_')[i]))
            seq_frame_loc_dict[frame_id] = point_list

        gt_seq_trans_dict[gt_seq_ind] = trans
        gt_seq_dict[gt_seq_ind] = seq_frame_loc_dict
        gt_seq_quality_dict[gt_seq_ind] = seq_frame_quality_dict

    return gt_seq_dict, gt_seq_quality_dict, gt_seq_trans_dict, gt_notcare_num

def load_pre_ata(track_seq_dict):
    """Loading predict data for ata

    Args:
        track_seq_dict (dict): track result

    Returns:
        dict(): each instance's bbox

    Returns:
        str: recognition word

    """
    track = track_seq_dict['track']

    if 'text' in track_seq_dict:
        cur_recog_word = track_seq_dict['text']

    # For pure track evaluation, we init all recog as '555'
    else:
        cur_recog_word = '555'

    data = list()
    seq_frame_loc_dict = {}

    # If quality score do not exist, max_score will return 1
    if 'scores' in track_seq_dict:
        scores = track_seq_dict['scores']

    for idx, instance in enumerate(track):
        img_frame_id = int(instance.split(',')[0])
        point_vec = instance.split(',')[1].split('_')
        recog_word = instance.split(',')[-1]
        track_point_list = []

        for i in range(8):
            track_point_list.append(np.int(point_vec[i]))
        seq_frame_loc_dict[img_frame_id] = track_point_list

    return seq_frame_loc_dict, cur_recog_word


def load_gt_mot(gt_dict, voca_list):
    """Loading gt data for mot

    Args:
        gt_dict (dict): gt data
        voca_list: vocabulary list

    Returns:
       int: video first frame number

    Returns:
       int: video last frame number

    Returns:
       dict(): each frame's predict bboxes

    Returns:
       dict(): each track's recognition result

    Returns:
       dict(): the care dict for each bbox

    """
    video_start_frame = 1e7
    video_end_frame = -1

    # 3 dict used for evaluation
    gt_frame_bbox_dict = {}
    gt_seq_trans_dict = {}
    gt_frame_care_dict = {}

    for gt_seq_ind in gt_dict:
        trans = gt_dict[gt_seq_ind]['trans']
        
        if voca_list:
            trans = process_gt_trans(trans, voca_list)

        gt_seq_trans_dict[gt_seq_ind] = trans

        for instance in gt_dict[gt_seq_ind]['track']:

            # Extract key infomation
            frame_id = int(instance.split(',', 2)[0])
            instance_trans = instance.split(',', 1)[1].rsplit(',', 2)[0]
            quality = instance.rsplit(',', 2)[-2]

            # Calculate the first and last frame id
            if video_start_frame > frame_id:
                video_start_frame = int(frame_id)
            if video_end_frame < frame_id:
                video_end_frame = int(frame_id)

            # Extract bbox
            point_list = list()
            point_list.append(gt_seq_ind)
            bbox = instance.rsplit(',', 2)[-1]
            for i in range(8):
                point_list.append(np.int(bbox.split('_')[i]))
        
            if frame_id not in gt_frame_bbox_dict.keys():
                tmp_pointlist_list = list()
                tmp_pointlist_list.append(point_list)
                gt_frame_bbox_dict[frame_id] = tmp_pointlist_list
            else:
                gt_frame_bbox_dict[frame_id].append(point_list)

            # Record if this instance is care or not
            care_dict = {}
            if gt_seq_trans_dict[gt_seq_ind] == '###' or instance_trans == '###' or quality == 'LOW' or len(instance_trans) < 3:
                care_dict[gt_seq_ind] = 0
            else:
                care_dict[gt_seq_ind] = 1

            if frame_id not in gt_frame_care_dict.keys():
                tmp_care_list = list()
                tmp_care_list.append(care_dict)
                gt_frame_care_dict[frame_id] = tmp_care_list
            else:
                gt_frame_care_dict[frame_id].append(care_dict)

    return video_start_frame, video_end_frame, gt_frame_bbox_dict, gt_seq_trans_dict, gt_frame_care_dict


def load_pre_mot(track_seq_dict, prob_ind):
    """Loading predict data for mot

    Args:
        track_seq_dict (dict): track result
        prob_ind (int): the track seq id

    Returns:
        dict(): each instance's bbox

    Returns:
        str: recognition word

    """

    track = track_seq_dict['track']

    seq_frame_loc_dict = {}

    if 'text' in track_seq_dict:
        cur_recog_word = track_seq_dict['text']
        
    # For pure track evaluation, we init all recog as '555'
    else:
        cur_recog_word = '555'
    
    for idx, instance in enumerate(track):
        img_frame_id = int(instance.split(',')[0])
        point_vec = instance.split(',')[1].split('_')

        track_point_list = list()
        track_point_list.append(prob_ind)
        for i in range(8):
            track_point_list.append(np.int(point_vec[i]))
        seq_frame_loc_dict[img_frame_id] = track_point_list


    return seq_frame_loc_dict, cur_recog_word


def load_gt_fscore(gt_dict, voca_list):
    """Loading gt data for fscore

    Args:
        gt_dict (dict): gt data
        voca_list: vocabulary list

    Returns:
       dict(): record each track seq by gt trans

    Returns:
       dict(): record the bboxes of all frames in each track seq

    Returns:
       dict(): record the seq id in gts whether has been matched

    Returns:
       dict(): record each track seq's final recognition result

    Returns:
       int:  record the number of ignore gt nums, like low quality

    """

    # 'gt_word':[seq_idx1,seq_idx2,..]
    label_index_dict = {}

    # Record the bboxes of all frames in each track seq
    index_loc_dict = {}

    # Record the seq id in gts whether has been matched
    gt_matched_dict = {}

    # Record texts of all frames in each track seq
    gt_seq_trans_dict = {}

    # Record the ignore gt nums, like low quality
    gt_notcare_num = 0

    for gt_seq_ind in gt_dict:

        trans = gt_dict[gt_seq_ind]['trans']

        if voca_list:
            trans = process_gt_trans(trans, voca_list)

        gt_seq_trans_dict[gt_seq_ind] = trans

        if trans == '###':
            gt_notcare_num += 1

        if gt_seq_ind in gt_matched_dict.keys():
            print('gt seq id not unique, there exists two seq has same id')

        gt_matched_dict[gt_seq_ind] = 0
        if trans not in label_index_dict.keys():
            label_index_dict[trans] = []

        label_index_dict[trans].append(gt_seq_ind)

        frame_loc_dict = {}

        for instance in gt_dict[gt_seq_ind]['track']:
            frame_id = np.int(instance.split(',')[0])
            bbox = instance.rsplit(',', 1)[-1]
            point_list = []
            for i in range(8):
                point_list.append(bbox.split('_')[i])
            frame_loc_dict[frame_id] = point_list

        index_loc_dict[gt_seq_ind] = frame_loc_dict

    return label_index_dict, index_loc_dict, gt_matched_dict, gt_seq_trans_dict, gt_notcare_num


def load_pre_fscore(track_seq_dict):
    """Loading predict data for F-score

    Args:
        track_seq_dict (dict): track result

    Returns:
        dict(): each instance's bbox

    Returns:
        int: selected frame id to stand for this track

    Returns:
        str: recognition word

    """
    cur_recog_word = track_seq_dict['text']
    tracks = track_seq_dict['track']

    seq_frame_loc_dict = {}
    selected_frame_ind = -1
    max_score = -1

    # If you have quality scores, we select the recog word and frame by scores
    if 'scores' in track_seq_dict.keys():
        scores = track_seq_dict['scores']

        # Iter each track in all tracks of one video
        for idx, track in enumerate(tracks):
            img_frame_id = int(track.split(',')[0])
            point_vec = track.split(',')[1].split('_')
            recog_word = track.split(',')[-1]
            track_point_list = []
            score = scores[idx]

            if score > max_score:
                max_score = score
                selected_frame_ind = img_frame_id

            for i in range(8):
                track_point_list.append(np.int(point_vec[i]))
            seq_frame_loc_dict[img_frame_id] = track_point_list


    # Or you must define a selected frame
    elif 'select_frame' in track_seq_dict.keys():

        selected_frame_ind = int(track_seq_dict['select_frame'].split(',')[0])
        for idx, track in enumerate(tracks):
            img_frame_id = int(track.split(',')[0])
            point_vec = track.split(',')[1].split('_')
            track_point_list = []
            for i in range(8):
                track_point_list.append(np.int(point_vec[i]))
            seq_frame_loc_dict[img_frame_id] = track_point_list

    else:
        raise KeyError("You neither have no 'scores' or 'select_frame' in keys, which can not do F-score evaluation")

    return seq_frame_loc_dict, selected_frame_ind, cur_recog_word
