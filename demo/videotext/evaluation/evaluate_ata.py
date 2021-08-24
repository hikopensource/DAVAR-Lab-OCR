# -*- coding: utf-8 -*-
import os
import json
import time

import numpy as np

import evaluate_utils


def ATA(track_dict, gt_dict, voca_dict, care_rcg):
    """Evaluation for MOT

    Args:
        track_dict(dict): track results
        gt_dict(dict): gt results
        voca_dict(str): vocabulary directory
        care_rcg(bool): whether to add recognition result constraint

    """
    video_num = 0
    avg_video_ata = 0

    for video_name in track_dict.keys():

        video_num = video_num + 1
        print('processing ' + video_name)

        # Loading vocabulary, If no vocabulary, the list will be empty
        if voca_res is not None:
            voca_list = voca_dict[video_name]
        else:
            voca_list = list()
        
        # Loading gt info for evaluation
        gt_seq_dict, gt_seq_quality_dict, gt_seq_trans_dict, gt_notcare_num = evaluate_utils.load_gt_ata(
            gt_dict[video_name], voca_list)

        # Construct tracked (tested) seq result dict
        track_seq_dict = {}
        track_seq_word_dict = {}

        for prob_ind in track_dict[video_name].keys():

            # Loading predict track info
            seq_frame_loc_dict, cur_recog_word = evaluate_utils.load_pre_ata(track_dict[video_name]
                                                                                        [prob_ind])
            
            cur_recog_word = cur_recog_word.upper()

            track_seq_word_dict[prob_ind] = cur_recog_word
            track_seq_dict[prob_ind] = seq_frame_loc_dict

        # Matching: implement sequence mapping using hungarian with overlapping maximized
        tracked_num = len(track_seq_dict.keys())
        gt_num = len(gt_seq_dict)

        # Square match matrix, which save the number of match with care and recognition is correct pairs
        match_without_low_matrix = np.zeros([max(tracked_num, gt_num), max(tracked_num, gt_num)], np.float)

        # IOU matrix,
        iou_matrix = np.zeros([max(tracked_num, gt_num),max(tracked_num, gt_num)],np.float)

        for track_idx_num in range(len(track_seq_dict.keys())):

            # Get current predict track seq
            track_idx = list(track_seq_dict.keys())[track_idx_num]
            cur_track_frame_loc_dict = track_seq_dict[track_idx]

            # Iter all gt track seqs
            for gt_idx_num in range(len(gt_seq_dict.keys())):
                gt_idx = list(gt_seq_dict.keys())[gt_idx_num]

                # Current gt track seq's text bboxes
                cur_gt_frame_loc_dict = gt_seq_dict[gt_idx]

                # Current gt track seq's texts quality
                cur_gt_frame_quality_dict = gt_seq_quality_dict[gt_idx]

                # Current gt track seq's recognition result
                trans = gt_seq_trans_dict[gt_idx]

                match_without_low_num = 0
                spatio_temporal_iou = 0

                # Calculating those frames which exists in gt track seq
                for frame_id in cur_track_frame_loc_dict.keys():
                    if frame_id in cur_gt_frame_loc_dict.keys():

                        # Calculating IOU
                        track_poly = evaluate_utils.polygon_from_points(cur_track_frame_loc_dict[frame_id])
                        gt_poly = evaluate_utils.polygon_from_points(cur_gt_frame_loc_dict[frame_id])
                        iou = evaluate_utils.get_intersection_over_union(track_poly, gt_poly)

                        spatio_temporal_iou = spatio_temporal_iou + iou

                        # Condition that the iou is greater than threshold, gt bbox is not low quality and recognition
                        # result is correct, then the match_without_low_recog_num will add 1.
                        if iou >= 0.5:
                            if not cur_gt_frame_quality_dict[frame_id] == 'LOW' and not trans == '###':

                                if care_rcg:
                                    if track_seq_word_dict[track_idx] == gt_seq_trans_dict[gt_idx]:
                                        match_without_low_num = match_without_low_num + 1
                                else:
                                    match_without_low_num = match_without_low_num + 1

                iou_matrix[track_idx_num, gt_idx_num] = spatio_temporal_iou
                match_without_low_matrix[track_idx_num, gt_idx_num] = match_without_low_num

        # Padding the matrix into N*N for hungarian algorithm
        row_ind, col_ind = evaluate_utils.Hungary(match_without_low_matrix)

        # Get imaginary index
        gt_fake_index_list = []
        tracked_fake_index_list = []

        # Those match with imaginary gt are false positives
        if tracked_num > gt_num:
            for i in range(gt_num, tracked_num):
                gt_fake_index_list.append(i)

        # Those imaginary tracked res are miss
        else:
            for i in range(tracked_num, gt_num):
                tracked_fake_index_list.append(i)

        cur_seq_stda = 0
        not_care_matched_num = 0

        for match_idx in range(len(row_ind)):
            #
            if row_ind[match_idx] in tracked_fake_index_list:
                continue
            if col_ind[match_idx] in gt_fake_index_list:
                continue
            track_probe_seq_ind = list(track_seq_dict.keys())[row_ind[match_idx]]
            track_frame_loc_dict = track_seq_dict[track_probe_seq_ind]
            gt_seq_ind = list(gt_seq_dict.keys())[col_ind[match_idx]]

            # For gt not care case, only when iou is greater than 0.1 will be take as ture not care
            if gt_seq_trans_dict[gt_seq_ind] == '###':
                iou = np.max(iou_matrix[row_ind[match_idx], :])
                if iou >= 0.1:
                    not_care_matched_num = not_care_matched_num + 1
                continue

            gt_frame_loc_dict = gt_seq_dict[gt_seq_ind]

            # Remove the num matched with not care gt num ('LOW')
            track_withoutlow_frame_list = []
            gt_withoutlow_frame_list = []
            gt_frame_quality_dict = gt_seq_quality_dict[gt_seq_ind]

            # Remove the gt bbox which quality is low
            for frame_id in gt_frame_loc_dict.keys():
                if not gt_frame_quality_dict[frame_id] == 'LOW':
                    gt_withoutlow_frame_list.append(frame_id)

            # Remove the track bbox which matched gt box quality is low
            for frame_id in track_frame_loc_dict.keys():
                if frame_id in gt_frame_loc_dict.keys():
                    if not gt_frame_quality_dict[frame_id] == 'LOW':
                        track_withoutlow_frame_list.append(frame_id)
                else:
                    track_withoutlow_frame_list.append(frame_id)

            # Calculating the union number of frames
            union_frame_num = len(list(set(track_withoutlow_frame_list).union(set(gt_withoutlow_frame_list))))

            # Calculating ATA
            cur_seq_stda = cur_seq_stda + np.float(
                match_without_low_matrix[row_ind[match_idx], col_ind[match_idx]]) / union_frame_num

        # Calculating the video-level ata
        cur_video_ata = np.float(cur_seq_stda) / (np.float(
            gt_num - gt_notcare_num + tracked_num - not_care_matched_num) / 2)

        print('current video ATA : '+str(cur_video_ata)+'\n')
        avg_video_ata = avg_video_ata + cur_video_ata

    final_ata = np.float(avg_video_ata) / video_num
    print("\n****************************final ATA result****************************************************")
    print('final ata : ' + str(final_ata))
    print("************************************************************************************************\n")
    return final_ata


if __name__ == '__main__':

    args = evaluate_utils.parse_args()
    start_time = time.time()

    # Define predict file
    with open(args.predict_file, 'r') as p_f:
        track_res = json.load(p_f)

    # Define gt
    with open(args.gt_file, 'r') as g_f:
        gt_res = json.load(g_f)

    # Define vocabulary
    if args.voca_file is not None:
        with open(args.voca_file, 'r') as v_f:
            voca_res = json.load(v_f)
    else:
        voca_res = None

    # end-to-end evaluation
    ATA(track_res, gt_res, voca_res, args.care_rcg)

    end_time = time.time()
    
    print('Running time: %s Seconds' % (end_time - start_time))
