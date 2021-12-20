# -*- coding: utf-8 -*-
"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    filter.py
# Abstract       :    postprocessing for filtering low quality sequences

# Current Version:    1.0.0
# Date           :    2021-07-15
##################################################################################################
"""
import os
import argparse
import time
import json

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
    parser.add_argument('input_file', help='predict results')
    parser.add_argument('output_file', help='filtered predict results')
    parser.add_argument('--voca_file', type=str, default=None)

    args_ = parser.parse_args()
    return args_


def filter_operate(seq_frame_loc_dict, max_score, cur_recog_word, voca_list):
    """Filter operation to filter out low quality track seq

    Args:
        seq_frame_loc_dict (dict): each instance's bbox
        max_score (float): highest quality score
        cur_recog_word : recognition result
        voca_list: vocabulary list

    Returns:
       bool: whether to filter out this track

    Returns:
       str: correct recognition result

    """
    cur_recog_word = cur_recog_word.upper()

    filter_flag = False

    # 0. filter by max_score
    filter_score_thresh = 0.78

    if max_score < filter_score_thresh:
        filter_flag = True
        return filter_flag, None


    # 2.  filter by short length of track seq
    if len(seq_frame_loc_dict.keys()) <= 5:
        filter_flag = True
        return filter_flag, None

    # 3.  filter by short length of recognition word length
    if len(cur_recog_word) < 3:
        filter_flag = True
        return filter_flag, None

    # 4. find nearest match in vocabulary
    contain_number = False

    for i in range(len(cur_recog_word)):
        cur_index = string_map.index(cur_recog_word[i])
        if 0 <= cur_index <= 9:
            contain_number = True

    # If vocabulary exists, using vocabulary to correct recog word
    if voca_list:
        min_recog_dist = 1e7
        recog_word = ''
        if not contain_number:
            for voca in voca_list:
                cur_dist = Levenshtein.distance(voca, cur_recog_word)
                if cur_dist < min_recog_dist:
                    min_recog_dist = cur_dist
                    recog_word = voca
                if cur_dist == min_recog_dist:
                    if len(recog_word) < len(voca):
                        recog_word = voca
        else:
            recog_word = cur_recog_word


        # 5. filter large edit dist to voca
        if not contain_number and min_recog_dist >= (len(cur_recog_word) + 1) / 3:
            filter_flag = True
            return filter_flag, None

    # Using original word
    else:
        recog_word = cur_recog_word

    return filter_flag, recog_word


def load_data(track_seq_dict):
    """Loading predict data for ata

    Args:
        track_seq_dict (dict): track result

    Returns:
        dict(): each instance's bbox

    Returns:
        str: recognition word

    Returns:
        float: highest quality score

    """
    cur_recog_word = track_seq_dict['text']
    track = track_seq_dict['track']

    data = list()
    seq_frame_loc_dict = {}
    max_score = -1

    # If quality score do not exist, max_score will return 1
    if 'scores' in track_seq_dict:
        scores = track_seq_dict['scores']

    for idx, instance in enumerate(track):
        img_frame_id = int(instance.split(',')[0])
        point_vec = instance.split(',')[1].split('_')
        recog_word = instance.split(',')[-1]
        track_point_list = []

        if 'scores' in track_seq_dict:
            score = scores[idx]
            if score > max_score:
                max_score = score
            data.append([score, recog_word])

        for i in range(8):
            track_point_list.append(np.int(point_vec[i]))
        seq_frame_loc_dict[img_frame_id] = track_point_list

    if 'scores' in track_seq_dict:
        data = sorted(data, key=lambda x: float(x[0]), reverse=True)
        if len(data) >= 5:
            for i in range(5):
                if len(cur_recog_word) < len(data[i][1]):
                    cur_recog_word = data[i][1]
    else:
        max_score = 1.

    return seq_frame_loc_dict, cur_recog_word, max_score


if __name__ == "__main__":

    args = parse_args()
    start_time = time.time()

    # Define predict file
    with open(args.input_file, 'r') as p_f:
        track_res = json.load(p_f)

    # Define vobulary file
    if args.voca_file is not None:
        with open(args.voca_file, 'r') as v_f:
            voca_res = json.load(v_f)
    else:
        voca_res = None
    
    # Saving the filtered result
    filtered_result = dict()

    for video in track_res:
        
        filtered_result[video] = dict()

        # Loading vocabulary list
        if voca_res is not None:
            voca_list = voca_res[video]
        else:
            voca_list = list()

        # Itering on each track in a video
        for track_id in track_res[video]:

            # Loading track info
            seq_frame_loc_dict, cur_recog_word, max_score = load_data(track_res[video][track_id])

            # Filtering low quality track
            filter_flag, recog_word = filter_operate(seq_frame_loc_dict, max_score, cur_recog_word, voca_list)
            
            # If filter flag is True, filter this track seq
            if filter_flag:
                continue

            track_res[video][track_id]['text'] = recog_word.upper()
            filtered_result[video][track_id] = track_res[video][track_id]
    
    with open(args.output_file, 'w') as w_f:
        json.dump(filtered_result, w_f, indent = 4)

