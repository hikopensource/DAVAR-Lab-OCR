# -*- coding: utf-8 -*-
import os
import json
import time

import numpy as np

import evaluate_utils


def MOT(track_dict, gt_dict, voca_dict, care_rcg):
    """Evaluation for MOT

    Args:
        track_dict(dict): track results
        gt_dict(dict): gt results
        voca_dict(str): vocabulary directory
        care_rcg(bool): whether to add recognition result constraint

    """
    # The number of id switch times
    total_idswitch = 0

    # The number of TP
    total_truepos = 0

    # The number of FN
    total_falseneg = 0

    # The number of FP
    total_falsepos = 0

    # The sum of TP's iou
    total_iou = 0

    # The number of gt
    total_gt = 0

    # MOTP, MOTA
    TOTAL_MOTP = 0
    TOTAL_MOTA = 0

    for video_name in track_dict.keys():

        # The number of id switch times in a video
        idswitch = 0

        # The number of TP in a video
        truepos = 0

        # The number of FN in a video
        falseneg = 0

        # The number of FP in a video
        falsepos = 0

        # The sum of iou in a video
        iou = 0

        # The number of gt in a video
        gt = 0

        # Previous frame mapping dict
        premapping = {}

        print('processing ' + video_name)

        # Loading vocabulary, If no vocabulary, the list will be empty
        if voca_res is not None:
            voca_list = voca_dict[video_name]
        else:
            voca_list = list()

        # Loading gt info for evaluation
        video_start_frame, video_end_frame, gt_frame_bbox_dict, gt_seq_trans_dict, gt_frame_care_dict =\
            evaluate_utils.load_gt_mot(gt_dict[video_name], voca_list)


        # Record each frame's all predict text bbox
        track_frame_bbox_dict = {}

        # Record each track's final recognition result
        track_seq_word_dict = {}

        # Iter all track seq in a video
        for prob_ind in track_dict[video_name].keys():

            # Loading predict track info
            seq_frame_loc_dict, cur_recog_word = \
                evaluate_utils.load_pre_mot(track_dict[video_name][prob_ind], prob_ind)

            cur_recog_word = cur_recog_word.upper()

            # Classify each text bbox into corresponding frame id
            for frame_idx in seq_frame_loc_dict.keys():
                if not track_frame_bbox_dict.get(frame_idx):
                    tmp_pointlist_list = list()
                    tmp_pointlist_list.append(seq_frame_loc_dict[frame_idx])
                    track_frame_bbox_dict[frame_idx] = tmp_pointlist_list
                else:
                    track_frame_bbox_dict[frame_idx].append(seq_frame_loc_dict[frame_idx])

            track_seq_word_dict[prob_ind] = cur_recog_word

        # Calculating id switch, FN, FP, TP by iter from first frame to last frame in a video
        for frame_idx in range(video_start_frame, video_end_frame + 1):

            # Get each frame det res
            cur_frame_track_res = []
            cur_frame_gt = []
            cur_frame_gt_care = []
            if frame_idx in track_frame_bbox_dict.keys():
                cur_frame_track_res = track_frame_bbox_dict[frame_idx]
            if frame_idx in gt_frame_bbox_dict.keys():
                cur_frame_gt = gt_frame_bbox_dict[frame_idx]
                cur_frame_gt_care = gt_frame_care_dict[frame_idx]

            # If current frame has no instance, init pre mapping dict
            if len(cur_frame_track_res) == 0 and len(cur_frame_gt) == 0:
                premapping = {}
                continue

            # Calculating current frame's care gt number
            cur_frame_care_gt_num = 0
            for care_dict in cur_frame_gt_care:
                for key in care_dict.keys():
                    if care_dict[key] == 1:
                        cur_frame_care_gt_num = cur_frame_care_gt_num + 1
            gt = gt + cur_frame_care_gt_num
            tracked_num = len(cur_frame_track_res)
            gt_num = len(cur_frame_gt)
            iou_matrix = np.zeros([max(tracked_num, gt_num), max(tracked_num, gt_num)], np.float)

            # Calculating current frame's gt bboxes and predict bboxes iou matrix
            for track_idx in range(tracked_num):
                track_point_list = list()
                for i in range(8):
                    track_point_list.append(cur_frame_track_res[track_idx][i + 1])
                track_poly = evaluate_utils.polygon_from_points(track_point_list)
                for gt_idx in range(gt_num):
                    gt_point_list = list()
                    for i in range(8):
                        gt_point_list.append(cur_frame_gt[gt_idx][i + 1])
                    gt_poly = evaluate_utils.polygon_from_points(gt_point_list)
                    cur_iou = evaluate_utils.get_intersection_over_union(track_poly, gt_poly)
                    iou_matrix[track_idx, gt_idx] = cur_iou
            
            # Hungarian matching
            row_ind, col_ind = evaluate_utils.Hungary(iou_matrix)

            # Get imaginary index
            gt_fake_index_list = []
            tracked_fake_index_list = []

            # Those match with imaginary gt are FP
            if tracked_num > gt_num:
                for i in range(gt_num, tracked_num):
                    gt_fake_index_list.append(i)

            # Those imaginary tracked res are FN
            else:
                for i in range(tracked_num, gt_num):
                    tracked_fake_index_list.append(i)

            # Get current mapping [gt_id,track_id], the mapping dict contains all gt id and imaginary gt
            mapping = {}
            mapped_iou_dict = {}
            mapped_care_dict = {}
            for match_idx in range(len(row_ind)):

                # Those imaginary tracked res are FN
                if row_ind[match_idx] in tracked_fake_index_list:

                    gt_id = cur_frame_gt[col_ind[match_idx]][0]
                    if cur_frame_gt_care[col_ind[match_idx]][gt_id] == 1:
                        falseneg = falseneg + 1
                    continue

                # Those match with imaginary gt are FP
                if col_ind[match_idx] in gt_fake_index_list:
                    falsepos = falsepos + 1
                    continue

                # Get current mapping
                track_id = cur_frame_track_res[row_ind[match_idx]][0]
                gt_id = cur_frame_gt[col_ind[match_idx]][0]
                mapping[gt_id] = track_id
                mapped_iou_dict[gt_id] = iou_matrix[row_ind[match_idx], col_ind[match_idx]]
                mapped_care_dict[gt_id] = cur_frame_gt_care[col_ind[match_idx]][gt_id]

            # If current mapping and previous mapping all exist, calculating TP, FP, FN, id switch
            if len(mapping.keys()) > 0 and len(premapping.keys()) > 0:
                for i in range(len(mapping.keys())):
                    cur_gt_id = list(mapping.keys())[i]
                    cur_tracked_id = mapping[cur_gt_id]

                    # Calculating care case
                    if mapped_care_dict[cur_gt_id] == 1:

                        # Only if the iou is greater than threshold and recognition result is correct, can they be TP
                        if mapped_iou_dict[cur_gt_id] > 0.5:
                            
                            # The care_rcg flag is to decide on whether adding recognition constraint
                            if care_rcg:

                                if gt_seq_trans_dict[cur_gt_id] == track_seq_word_dict[cur_tracked_id]:
                                    truepos = truepos + 1
                                    iou = iou + mapped_iou_dict[cur_gt_id]

                                # If recognition result is not correct, take it as FP
                                else:
                                    falsepos = falsepos + 1

                            # If not care about rcg result, the pure MOT will be calculated
                            else:
                                truepos = truepos + 1
                                iou = iou + mapped_iou_dict[cur_gt_id]

                            # Check if there is id switch
                            if cur_gt_id in premapping.keys():
                                pre_matched_track_id = premapping[cur_gt_id]
                                if not pre_matched_track_id == mapping[cur_gt_id]:
                                    idswitch = idswitch + 1

                        # If iou is less than threshold, take it as FP
                        else:
                            falsepos = falsepos + 1

                    # Calculating not care case, the iou less than 0.1 will be take as FP
                    else:
                        if mapped_iou_dict[cur_gt_id] < 0.1:
                            falsepos = falsepos + 1

            # If only current mapping exists, do not calculate id switch
            elif len(mapping.keys()) > 0:
                for i in range(len(mapping.keys())):
                    cur_gt_id = list(mapping.keys())[i]
                    cur_tracked_id = mapping[cur_gt_id]

                    # Only if the iou is greater than threshold and recognition result is correct, can they be TP
                    if mapped_care_dict[cur_gt_id] == 1:
                        if mapped_iou_dict[cur_gt_id] > 0.5:
                            
                            if care_rcg:

                                if gt_seq_trans_dict[cur_gt_id] == track_seq_word_dict[cur_tracked_id]:
                                    truepos = truepos + 1
                                    iou = iou + mapped_iou_dict[cur_gt_id]

                                # If recognition result is not correct, take it as FP
                                else:
                                    falsepos = falsepos + 1

                            else:
                                truepos = truepos + 1
                                iou = iou + mapped_iou_dict[cur_gt_id]

                        # If iou is less than threshold, take it as FP
                        else:
                            falsepos = falsepos + 1

                    # Calculating not care case
                    else:
                        if mapped_iou_dict[cur_gt_id] < 0.1:
                            falsepos = falsepos + 1

            # If only pre mapping exists, use previous mapping as current mapping for next map
            elif len(premapping.keys()) > 0:
                mapping = premapping

            # Update mapping list
            for i in range(len(cur_frame_gt)):
                cur_gt_id = cur_frame_gt[i][0]
                if cur_gt_id not in mapping.keys():
                    if cur_gt_id in premapping.keys():
                        mapping[cur_gt_id] = premapping[cur_gt_id]

            # Save for next frame
            premapping = mapping

        total_iou = total_iou + iou
        total_truepos = total_truepos + truepos
        total_falseneg = total_falseneg + falseneg
        total_falsepos = total_falsepos + falsepos
        total_idswitch = total_idswitch + idswitch
        total_gt = total_gt + gt

        if truepos == 0:
            MOTP = 0.
        else:
            MOTP = np.float(iou) / truepos

        if gt == 0:
            MOTA = 1.
        else:
            MOTA = 1. - (np.float(falseneg + falsepos + idswitch) / gt)

        print(str(video_name)+' MOTP: '+str(MOTP))
        print(str(video_name) + ' MOTA: ' + str(MOTA)+'\n\n')
        TOTAL_MOTP = TOTAL_MOTP + MOTP
        TOTAL_MOTA = TOTAL_MOTA + MOTA

    if total_truepos == 0:
        TOTAL_MOTP = 0.
    else:
        TOTAL_MOTP = np.float(total_iou) / total_truepos

    TOTAL_MOTA = 1. - (np.float(total_falseneg + total_falsepos + total_idswitch) / total_gt)

    print("\n****************************final MOT result****************************************************")
    print('avg all frames MOTP: ' + str(TOTAL_MOTP))
    print('avg all frames MOTA: ' + str(TOTAL_MOTA))
    print("************************************************************************************************\n")


if __name__ == '__main__':


    args = evaluate_utils.parse_args()
    start_time = time.time()

    # Define predict file
    with open(args.predict_file, 'r') as p_f:
        track_res = json.load(p_f)

    # Define gt
    with open(args.gt_file, 'r') as g_f:
        gt_res = json.load(g_f)

    if args.voca_file is not None:
        with open(args.voca_file, 'r') as v_f:
            voca_res = json.load(v_f)
    else:
        voca_res = None

    # MOT evaluation
    MOT(track_res, gt_res, voca_res, args.care_rcg)

    end_time = time.time()
    
    print('Running time: %s Seconds' % (end_time - start_time))
