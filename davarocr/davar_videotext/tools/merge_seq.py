"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    merge_seq.py
# Abstract       :    postprocessing for merge same track sequences

# Current Version:    1.0.0
# Date           :    2021-07-15
##################################################################################################
"""
import os
import argparse
import json

import numpy as np
import mmcv

import test_utils


def parse_args():
    """

    Returns:
        args parameter of model test

    """
    parser = argparse.ArgumentParser(description='DavarOCR test video text track merge')
    parser.add_argument('config', help='test config file path')
    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':

    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    input_file = os.path.join(cfg.out_dir, cfg.out_file)
    output_file = os.path.join(cfg.merge_out_dir, cfg.merge_out_file)

    # The max interval to merge
    merge_max_interval = cfg.merge_max_interval

    # The tight merge iou threshold
    merge_thresh_tight = cfg.merge_thresh_tight

    # The loose merge iou threshold
    merge_thresh_loose = cfg.merge_thresh_loose

    # The tight edit distance iou threshold
    edit_dist_iou_thresh_tight = cfg.edit_dist_iou_thresh_tight

    # The loose edit distance iou threshold
    edit_dist_iou_thresh_loose = cfg.edit_dist_iou_thresh_loose

    frame_min_index = cfg.frame_min_index
    frame_max_index = cfg.frame_max_index
    max_constant = cfg.max_constant

    # Get original track results
    with open(input_file, 'r') as read_file:
        track_res = json.load(read_file)

    filter_track_res = dict()

    for video_key in track_res.keys():

        # Record the start and end frame bbox for all track sequences in a video
        seq_start_frame_dict = {}
        seq_end_frame_dict = {}
        seq_start_loc_dict = {}
        seq_end_loc_dict = {}

        # Record each track sequence's highest quality score
        seq_score_dict = {}

        # Record each track sequence's highest quality score frame number
        seq_selectedframe_dict = {}

        # Record the merged track id
        merge_key_dict = {}

        # Record each track  highest quality score recognition result
        seq_word_dict = {}

        # Record the track id which has highest quality score in a merged list
        seq_best_seq_dict = {}
        seq_point_dict = {}
        img_point_dict = {}

        print("processing video: ", video_key)

        # Get each track data in a video
        for track_key in track_res[video_key].keys():

            tracks = track_res[video_key][track_key]['track']
            scores = track_res[video_key][track_key]['scores']
            highest_index = scores.index(max(scores))
            highest_score = scores[highest_index]
            start_frame = frame_min_index
            end_frame = frame_max_index

            # Get each instances' bbox, sequence start, end frame id
            for instance in tracks:
                frame_id = int(instance.split(',')[0])
                bbox = instance.split(',')[1]
                recog_word = instance.split(',')[2:]
                recog_word = "".join(recog_word)
                point_vec = bbox.split('_')
                point_list = []
                for point in point_vec:
                    point_list.append(np.int(point))
                img_point_dict[frame_id] = point_list
                start_frame = min(frame_id, start_frame)
                end_frame = max(frame_id, end_frame)

            # Record each track start, end frame id, bbox
            seq_start_frame_dict[track_key] = start_frame
            seq_start_loc_dict[track_key] = img_point_dict[start_frame]
            seq_end_frame_dict[track_key] = end_frame
            seq_end_loc_dict[track_key] = img_point_dict[end_frame]

            # Judge valid
            recog_word_valid = False
            recog_score_valid = False
            highest_recog = tracks[highest_index].split(',')[2:]
            highest_recog = "".join(highest_recog)

            if len(highest_recog) > 0:
                recog_word_valid = True

            if recog_word_valid:
                highest_track = tracks[highest_index]
                frame_id = int(highest_track.split(',')[0])
                recog_word = highest_track.split(',')[2:]
                recog_word = "".join(recog_word)
                seq_selectedframe_dict[track_key] = frame_id
                seq_score_dict[track_key] = highest_score
                seq_point_dict[track_key] = img_point_dict
                merge_key_dict[track_key] = [track_key]
                seq_best_seq_dict[track_key] = track_key
                # here the word is chosen by the highest score recognized word, in substitution it can be selected by
                # filtering and majority voting
                seq_word_dict[track_key] = recog_word

            img_point_dict = {}

        # Merge operation when this video iter over existing sequences to find merged track: check until no track can be
        # found
        has_merge_flag = True

        # Init best seq dict
        for tmp_key in seq_word_dict:
            seq_best_seq_dict[tmp_key] = tmp_key

        # Loop until there is no track seq merged
        while has_merge_flag:
            has_merge_flag = False

            merged_flag_dict = {}
            key_list = []
            # Record all the track id
            for key in seq_word_dict:
                merged_flag_dict[key] = 0
                key_list.append(key)

            key_idx = 0
            merged_flag_dict[key_list[0]] = 1

            # Iter all the track id to find merge seq
            while key_idx < len(key_list):
                key1 = key_list[key_idx]

                if merged_flag_dict[key1] == 1:
                    key_idx += 1
                    continue
                merged_list = []
                for merge_key in merge_key_dict[key1]:
                    merged_list.append(merge_key)

                # Fetch the merged track's highest score and frame id, recognition result
                merged_score = seq_score_dict[key1]
                best_score_frameid = seq_selectedframe_dict[key1]
                merged_word = seq_word_dict[key1]

                # Loop until this track can not find any seq to merge
                this_merged = True
                while this_merged:
                    this_merged = False

                    # Fetch another track id to judge if can be merged
                    key2_idx = 0
                    while key2_idx < len(key_list):
                        key2 = key_list[key2_idx]
                        if key2 == key1:
                            key2_idx = key2_idx + 1
                            continue

                        # Calc edit dist iou
                        edit_dist_iou_val = test_utils.edit_dist_iou(seq_word_dict[key1], seq_word_dict[key2])

                        # Condition 1 to merge start_seq1 < start_seq2 < end_seq1 + max_inv'
                        if seq_start_frame_dict[key1] <= seq_start_frame_dict[key2] <= \
                                seq_end_frame_dict[key1] + merge_max_interval:
                            seq2_start_frame = seq_start_frame_dict[key2]
                            seq2_start_loc = seq_start_loc_dict[key2]
                            seq2_start_poly = test_utils.polygon_from_points(seq2_start_loc)

                            # Find nearest frame id in key1
                            key1_point_dict = seq_point_dict[key1]
                            nearest_frame = -1
                            frame_dist = max_constant
                            for key1_frame in key1_point_dict.keys():
                                if abs(seq2_start_frame - key1_frame) < frame_dist:
                                    nearest_frame = key1_frame
                                    frame_dist = abs(seq2_start_frame - key1_frame)
                            seq1_nearest_poly = test_utils.polygon_from_points(key1_point_dict[nearest_frame])

                            # Calc iou
                            iou = test_utils.get_intersection_over_union(seq2_start_poly, seq1_nearest_poly)

                            # Judge if satisfy the merge condition
                            if (iou >= merge_thresh_tight and edit_dist_iou_val >= edit_dist_iou_thresh_loose) or (
                                    iou >= merge_thresh_loose and edit_dist_iou_val >= edit_dist_iou_thresh_tight):

                                # Merge
                                this_merged = True
                                has_merge_flag = True
                                for merge_key in merge_key_dict[key2]:
                                    merged_list.append(merge_key)

                                # Update key2 into key1
                                if seq_end_frame_dict[key1] < seq_end_frame_dict[key2]:
                                    seq_end_frame_dict[key1] = seq_end_frame_dict[key2]
                                    seq_end_loc_dict[key1] = seq_end_loc_dict[key2]

                                # Update points and frame info
                                seq_point_dict[key1].update(seq_point_dict[key2])

                                # Using the highest quality score seq
                                if seq_score_dict[key1] < seq_score_dict[key2]:
                                    # Update
                                    seq_best_seq_dict[key1] = seq_best_seq_dict[key2]
                                    seq_selectedframe_dict[key1] = seq_selectedframe_dict[key2]
                                    seq_score_dict[key1] = seq_score_dict[key2]
                                    seq_word_dict[key1] = seq_word_dict[key2]

                                # Update highest quality score track id
                                best_score_frameid = seq_selectedframe_dict[key1]
                                merged_score = seq_score_dict[key1]
                                merged_word = seq_word_dict[key1]

                                # Output
                                print('seq ' + str(key1) + ' and ' + str(key2) + ' merged, iou: ' + str(
                                    iou) + ' edit_iou: ' + str(edit_dist_iou_val) + ' ' + str(
                                    seq_word_dict[key1]) + ' ' + str(seq_word_dict[key2]) + ' ' + str(
                                    nearest_frame) + ' ' + str(seq2_start_frame))

                                # Remove key2 val in dict
                                seq_word_dict.pop(key2)
                                key_list.remove(key2)
                            else:

                                key2_idx = key2_idx + 1

                        # Condition 2 to merge start_seq2 < start_seq1 < end_seq2 + max_inv'
                        elif seq_start_frame_dict[key2] <= seq_start_frame_dict[key1] <= \
                                seq_end_frame_dict[key2] + merge_max_interval:
                            seq1_start_frame = seq_start_frame_dict[key1]
                            seq1_start_loc = seq_start_loc_dict[key1]
                            seq1_start_poly = test_utils.polygon_from_points(seq1_start_loc)

                            # Find nearest frame id in key1
                            key2_point_dict = seq_point_dict[key2]
                            nearest_frame = -1
                            frame_dist = max_constant
                            for key2_frame in key2_point_dict.keys():
                                if abs(seq1_start_frame - key2_frame) < frame_dist:
                                    nearest_frame = key2_frame
                                    frame_dist = abs(seq1_start_frame - key2_frame)
                            seq2_nearest_poly = test_utils.polygon_from_points(key2_point_dict[nearest_frame])

                            # Calc iou
                            iou = test_utils.get_intersection_over_union(seq1_start_poly, seq2_nearest_poly)

                            # Judge if satisfy the merge condition
                            if (iou >= merge_thresh_tight and edit_dist_iou_val >= edit_dist_iou_thresh_loose) or (
                                    iou >= merge_thresh_loose and edit_dist_iou_val >= edit_dist_iou_thresh_tight):
                                # Merge
                                this_merged = True
                                has_merge_flag = True

                                for merge_key in merge_key_dict[key2]:
                                    merged_list.append(merge_key)

                                # Update key2 into key1
                                seq_start_frame_dict[key1] = seq_start_frame_dict[key2]
                                seq_start_loc_dict[key1] = seq_start_loc_dict[key2]
                                if seq_end_frame_dict[key1] < seq_end_frame_dict[key2]:
                                    seq_end_frame_dict[key1] = seq_end_frame_dict[key2]
                                    seq_end_loc_dict[key1] = seq_end_loc_dict[key2]

                                # Update points and frame info
                                seq_point_dict[key1].update(seq_point_dict[key2])

                                if seq_score_dict[key1] < seq_score_dict[key2]:

                                    # Update
                                    seq_best_seq_dict[key1] = seq_best_seq_dict[key2]
                                    seq_selectedframe_dict[key1] = seq_selectedframe_dict[key2]
                                    seq_score_dict[key1] = seq_score_dict[key2]
                                    seq_word_dict[key1] = seq_word_dict[key2]

                                best_score_frameid = seq_selectedframe_dict[key1]
                                merged_score = seq_score_dict[key1]
                                merged_word = seq_word_dict[key1]

                                print('seq ' + str(key1) + ' and ' + str(key2) + ' merged, iou: ' + str(
                                    iou) + ' edit_iou: ' + str(edit_dist_iou_val) + ' ' + str(
                                    seq_word_dict[key1]) + ' ' + str(seq_word_dict[key2]) + ' ' + str(
                                    seq1_start_frame) + ' ' + str(nearest_frame))

                                # Remove key2
                                seq_word_dict.pop(key2)
                                key_list.remove(key2)
                            else:
                                key2_idx = key2_idx + 1
                        else:
                            key2_idx = key2_idx + 1

                # Store merge dict
                if key1 not in merge_key_dict.keys():
                    merge_key_dict[key1] = merged_list
                else:
                    for new_merge_key in merged_list:
                        if new_merge_key not in merge_key_dict[key1]:
                            merge_key_dict[key1].append(new_merge_key)

                merged_flag_dict[key1] = 1
                key_idx = key_idx + 1

        # Sum up merge result
        for sum_key, value in seq_word_dict.items():
            best_score_frameid = seq_selectedframe_dict[sum_key]
            merged_score = seq_score_dict[sum_key]

            merged_word = value
            merged_list = merge_key_dict[sum_key]
            best_score_seq_ind = seq_best_seq_dict[sum_key]

            # Get location and frame id for merge sequence
            seq_frame_loc_dict = {}
            for track_id in merged_list:
                tracks = track_res[video_key][track_id]['track']
                scores = track_res[video_key][track_id]['scores']
                for instance in tracks:
                    img_frame_id = int(instance.split(',')[0])
                    points_vec = instance.split(',')[1].split('_')
                    track_point_list = []
                    for point in points_vec:
                        track_point_list.append(int(point))

                    seq_frame_loc_dict[img_frame_id] = track_point_list

            if video_key not in filter_track_res:
                filter_track_res[video_key] = dict()

            if sum_key not in filter_track_res[video_key]:
                filter_track_res[video_key][sum_key] = dict()
                filter_track_res[video_key][sum_key]['track'] = list()
                filter_track_res[video_key][sum_key]['scores'] = list()
                filter_track_res[video_key][sum_key]['text'] = list()

            # Merge the same track sequences
            merge_tracks = []
            merge_score = []
            frame_score = dict()
            for track_id in merged_list:
                tracks = track_res[video_key][track_id]['track']
                scores = track_res[video_key][track_id]['scores']
                merge_tracks += tracks

                for track_idx, instance in enumerate(tracks):
                    frame_id = int(instance.split(',')[0])
                    score = scores[track_idx]
                    frame_score[frame_id] = score

            # Make the frames in the track sequence be in the order of frame id
            merge_tracks = sorted(merge_tracks, key=lambda x: int(x.split(',')[0]))
            for instance in merge_tracks:
                frame_id = int(instance.split(',')[0])
                merge_score.append(frame_score[frame_id])

            filter_track_res[video_key][sum_key]['track'] = merge_tracks
            filter_track_res[video_key][sum_key]['scores'] = merge_score
            filter_track_res[video_key][sum_key]['text'] = merged_word

    # Save
    with open(output_file, 'w') as write_file:
        json.dump(filter_track_res, write_file, indent=4)
