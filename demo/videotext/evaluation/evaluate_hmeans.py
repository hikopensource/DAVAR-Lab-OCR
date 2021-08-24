# -*- coding: utf-8 -*-
import os
import json
import time

import numpy as np

import evaluate_utils


def Fscore(track_dict, gt_dict, voca_dict, hit_iou_thresh=0.5):
    """End-to-End evaluation for F-score

    Args:
        track_dict(dict): track results
        gt_dict(dict): gt results
        voca_dict(dict): vocabulary
        hit_iou_thresh(float): iou threshold

    """
    # The matched track seqs number of predict tracks
    all_matched_pre_num = 0

    # The number of predict tracks
    all_pre_num = 0

    # The matched track seqs number of gt tracks
    all_macthed_gt_num = 0

    # The number of gt tracks
    all_gt_num = 0

    for video_name in track_dict.keys():
        print('process ' + video_name)

        # Loading vocabulary, If no vocabulary, the list will be empty
        if voca_res is not None:
            voca_list = voca_dict[video_name]
        else:
            voca_list = list()

        # Loading gt info for evaluation
        label_index_dict, index_loc_dict, gt_matched_dict, gt_seq_trans_dict, gt_notcare_num = evaluate_utils.\
            load_gt_fscore(gt_dict[video_name], voca_list)
        
        # Record the predict track seq match status
        pre_matched_dict = {}

        for prob_ind in track_dict[video_name].keys():

            selected_seq_ind = prob_ind

            # Loading predict track info
            seq_frame_loc_dict, selected_frame_ind, ori_recog_word = evaluate_utils.load_pre_fscore(
                track_dict[video_name][prob_ind])

            cur_recog_word = ori_recog_word.upper()

            if selected_seq_ind in pre_matched_dict.keys():
                print('the predict seq id is not unique')

            # Init
            pre_matched_dict[selected_seq_ind] = 0
            selected_loc = seq_frame_loc_dict[selected_frame_ind]
            selected_word = cur_recog_word
            max_iou = -1.
            max_gt_idx = ''

            # Iter for find the match track seq
            for gt_label in label_index_dict.keys():
                idx_list = label_index_dict[gt_label]
                for gt_idx in idx_list:

                    # If gt already matched, continue
                    if gt_matched_dict[gt_idx] == 1 or gt_matched_dict[gt_idx] == 2:
                        continue

                    # If the selected frame index do not exists in gt track seq, continue
                    if selected_frame_ind not in index_loc_dict[gt_idx].keys():
                        continue

                    # Calculating IOU
                    tmp_gt_poly = index_loc_dict[gt_idx][selected_frame_ind]
                    tmp_selected_poly = evaluate_utils.polygon_from_points(selected_loc)
                    tmp_corres_gt_poly = evaluate_utils.polygon_from_points(tmp_gt_poly)
                    iou = evaluate_utils.get_intersection_over_union(tmp_selected_poly, tmp_corres_gt_poly)

                    # Saving the max IOU
                    if iou >= max_iou:
                        max_iou = iou
                        max_gt_idx = gt_idx

            # Do not match any gt track seq
            if max_gt_idx == '':
                continue

            # Match with not care gt track sequence
            if gt_seq_trans_dict[max_gt_idx] == '###':
                pre_matched_dict[selected_seq_ind] = 2
                gt_matched_dict[max_gt_idx] = 2
            else:
                # Matched IOU is less than hit_iou_thresh, invalid
                if max_iou < hit_iou_thresh:
                    continue

                # Matched IOU satisfy the threshold and recogntion result is correct, valid match
                if max_iou >= hit_iou_thresh and gt_seq_trans_dict[max_gt_idx] == selected_word:
                    pre_matched_dict[selected_seq_ind] = 1
                    gt_matched_dict[max_gt_idx] = 1

        # Calculating matched number
        matched_pre_num = 0
        matched_gt_num = 0
        matched_notcare_pre_num = 0

        for key in pre_matched_dict.keys():

            # Match valid case
            if pre_matched_dict[key] == 1:
                matched_pre_num = matched_pre_num + 1

            # Match with not care case
            if pre_matched_dict[key] == 2:
                matched_notcare_pre_num = matched_notcare_pre_num + 1

        for key in gt_matched_dict:
            if gt_matched_dict[key] == 1:
                matched_gt_num = matched_gt_num + 1

        # If gt care number is 0, the recall will be 1.
        if (len(gt_matched_dict.keys()) - gt_notcare_num) == 0:
            recall_ = 1.
        else:
            recall_ = np.float(matched_pre_num) / (len(gt_matched_dict.keys()) - gt_notcare_num)

        # If predict care number is 0, the preccision will be 0.
        if (len(pre_matched_dict.keys()) - matched_notcare_pre_num) == 0:
            precision_ = 0.
        else:
            precision_ = np.float(matched_gt_num) / (len(pre_matched_dict.keys()) - matched_notcare_pre_num)

        # Calculating video-level F-score
        if recall_ + precision_ == 0.:
            h_means_ = 0.

        else:
            h_means_ = 2 * recall_ * precision_ / (recall_ + precision_)

        print(video_name, 'recall = ' + str(matched_pre_num) + '/' +
              str((len(gt_matched_dict.keys()) - gt_notcare_num)) + ' = ' + str(recall_))

        print(video_name, 'precision = ' + str(matched_gt_num) + '/' +
              str((len(pre_matched_dict.keys()) - matched_notcare_pre_num)) + ' = ' + str(precision_))

        print(video_name, 'h-means = ' + str(h_means_))

        all_matched_pre_num = all_matched_pre_num + matched_pre_num
        all_macthed_gt_num = all_macthed_gt_num + matched_gt_num
        all_pre_num = all_pre_num + (len(pre_matched_dict.keys()) - matched_notcare_pre_num)
        all_gt_num = all_gt_num + (len(gt_matched_dict.keys()) - gt_notcare_num)

    # Calculating final F-score
    recall = np.float(all_macthed_gt_num) / all_gt_num
    precision = np.float(all_matched_pre_num) / all_pre_num
    h_means = 2 * recall * precision / (recall + precision)
    print("\n****************************final Fscore result****************************************************")
    print('total recall = ' + str(all_macthed_gt_num) + '/' + str(all_gt_num) + ' = ' + str(recall))
    print('total precision = ' + str(all_matched_pre_num) + '/' + str(all_pre_num) + ' = ' + str(precision))
    print('total h-means = ' + str(h_means))
    print("***************************************************************************************************\n")


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

    # end-to-end evaluation
    Fscore(track_res, gt_res, voca_res)

    end_time = time.time()
    
    print('Running time: %s Seconds' % (end_time - start_time))
