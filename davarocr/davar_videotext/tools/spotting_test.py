"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    track_test.py
# Abstract       :    generate track result from detection result

# Current Version:    1.0.0
# Date           :    2021-07-10
##################################################################################################
"""


import os
import json
import argparse

import numpy as np
import torch

import mmcv
from mmcv.parallel import collate, scatter, MMDataParallel
from mmcv.runner import load_checkpoint
from sklearn.metrics.pairwise import cosine_similarity

from mmdet.datasets.pipelines import Compose
from davarocr.davar_rcg.models.builder import build_recognizor
import test_utils


def parse_args():
    """

    Returns:
        args parameter of model test

    """
    parser = argparse.ArgumentParser(description='DavarOCR test video text e2e')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--local_rank', type=int, default=0)
    args_ = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args_.local_rank)
    return args_


if __name__ == '__main__':

    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    # Set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.data.test.test_mode = True
    test_pipeline = Compose(cfg.test_pipeline)
    model_path = cfg.ckpts[0]['ModelPath']
    if not os.path.exists(model_path):
        print(model_path + ' not exist.')

    config_cfg = mmcv.Config.fromfile(cfg.ckpts[0]['ConfigPath'])
    test_cfg = config_cfg.test_cfg

    # Build recognition model
    model = build_recognizor(config_cfg.model, train_cfg=None, test_cfg=test_cfg)

    # Load the model pth file
    checkpoint = load_checkpoint(model, model_path, map_location='cpu')

    model.CLASSES = None

    model = MMDataParallel(model, device_ids=[0])
    device = next(model.parameters()).device
    model.eval()

    # Hyper parameters

    # The feature similarity threshold
    feat_sim_thresh = cfg.feat_sim_thresh

    # The feature similarity with adjacent threshold
    feat_sim_with_loc_thresh = cfg.feat_sim_with_loc_thresh

    # The track instance max exist duration
    max_exist_duration = cfg.max_exist_duration

    # The feature channel for tracking
    feat_channels = cfg.feat_channels

    # Constant eps
    eps = cfg.eps

    # The unique identification of text sequence in a video
    text_id = 0

    # The predicted detection result file by detection model
    ori_det_data = mmcv.load(cfg.testsets[0]["AnnFile"])

    img_prefix = cfg.testsets[0]["FilePre"]

    det_data = dict()

    # extract video and img name as key to save track result
    for key in ori_det_data.keys():
        video = key.split('/')[-2]
        frame_id = key.split('/')[-1]
        img_key = video + '/' + frame_id
        if video not in det_data.keys():
            det_data[video] = dict()
        det_data[video][img_key] = ori_det_data[key]

    # to save track sequence for  all videos
    track_res_dict = dict()

    # output(json) file to save track result
    out_dir = cfg.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # generate track sequence by video
    # step1 : construct history dict to store: text id (store folder name) / past duration / newest feat
    # step2 : extract text feat from current frame
    # step3 : get current matching list
    # step4 : confirm each match pair valid and updating feat and duration

    for video, value in det_data.items():
        print('processing video: ' + str(video))
        frame_nums = len(value.keys())

        # History data to match current data
        his_textid_list = []
        his_duration_list = []
        his_loc_list = []
        his_feat_array = None

        # To save track sequence for specific video
        track_res_dict[video] = dict()

        # The frame id in video should start from "1" and should be consecutive by default
        for frame_id in range(1, frame_nums + 1):
            print("processing frame :", frame_id)
            key = video + '/' + str(frame_id) + '.jpg'
            img_info = value[key]

            # Read predict bboxes from one image into list
            instance_infos = test_utils.instance_to_list(img_info, key)

            cur_frame_pos = []
            track_feature = None

            # Data pipelines and model output
            if len(instance_infos) > 0:
                batch_data = []
                for instance in instance_infos:
                    cur_frame_pos.append(instance['ann']['bbox'])
                    data = dict(img_info=instance, img_prefix=img_prefix)
                    data = test_pipeline(data)
                    batch_data.append(data)
                data_collate = collate(batch_data, samples_per_gpu=len(batch_data))
                device = int(str(device).rsplit(':', maxsplit=1)[-1])
                data = scatter(data_collate, [device])[0]

                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data)

                # Get model output
                texts = result['text']
                scores = result['scores']
                scores = scores.cpu().numpy()
                scores = scores.reshape(-1)
                track_feature = result['track_feature']
                track_feature = track_feature.cpu().numpy()
            else:
                texts = None
                scores = None
                track_feature = None

            # In the begging, when no history data to match, create all bboxes as new track sequences
            if len(his_textid_list) == 0:
                if len(instance_infos) > 0:
                    first_img_flag = True
                    for text_img_idx, instance in enumerate(instance_infos):

                        # Append the text id, duration time, bbox for a new text instance
                        his_textid_list.append(text_id)
                        his_duration_list.append(0)
                        his_loc_list.append(instance['ann']['bbox'])

                        # For first instance we need to reshape the history feature array to (1, feature_channel)
                        if first_img_flag:
                            his_feat_array = np.array(track_feature[text_img_idx, :])
                            first_img_flag = False

                        # Append left features
                        else:
                            his_feat_array = np.row_stack((his_feat_array, np.array([track_feature[text_img_idx, :]])))

                        # Update a new track
                        test_utils.update_track(text_id, video, track_res_dict, frame_id, instance['ann']['bbox'],
                                                texts[text_img_idx], scores[text_img_idx].item())

                        # Update text id for next track
                        text_id = text_id + 1
                continue

            ori_his_len = len(his_textid_list)
            his_matched_matrix = np.zeros([ori_his_len], np.int)

            # If current img has no text instance
            if len(instance_infos) == 0:

                # Update history data
                his_textid_list, his_duration_list, his_loc_list, his_feat_array = test_utils.\
                    update_history(his_textid_list, his_duration_list, his_loc_list, his_feat_array, his_matched_matrix,
                                   max_exist_duration, ori_his_len)

                continue

            # For the case that there is only one history feature, we should reshape to it (1, feat_channels)
            if len(his_feat_array.shape) == 1 and his_feat_array.shape[0] == feat_channels:
                his_feat_array = his_feat_array.reshape(1, his_feat_array.shape[0])

            #
            max_num = max(len(track_feature), len(his_feat_array))

            # Calc feature similarty, iou, adjacent matrix to match history track, In YORO, we only use feat_sim_matrix
            # and adja_maxtrix to match, You can adjust to your only task
            feat_sim_matrix = cosine_similarity(track_feature, his_feat_array)
            iou_matrix = np.zeros([len(cur_frame_pos), len(his_loc_list)], np.float)
            adja_matrix = np.zeros([len(cur_frame_pos), len(his_loc_list)], np.int)

            for cur_idx, cur_bbox in enumerate(cur_frame_pos):
                cur_poly = test_utils.polygon_from_points(cur_bbox)
                for his_idx, his_bbox in enumerate(his_loc_list):

                    # Calculate IOU
                    his_poly = test_utils.polygon_from_points(his_bbox)
                    cur_iou = test_utils.get_intersection_over_union(cur_poly, his_poly)
                    iou_matrix[cur_idx, his_idx] = cur_iou

                    # Calculate the expand coordinates
                    expand_start_x, expand_end_x, expand_start_y, expand_end_y = test_utils.calculate_expand(his_bbox)

                    for i in range(4):
                        if expand_start_x <= cur_frame_pos[cur_idx][2 * i] <= expand_end_x and \
                                expand_start_y <= cur_frame_pos[cur_idx][2 * i + 1] <= expand_end_y:
                            adja_matrix[cur_idx, his_idx] = 1

            # Aggregate feature similarity matrix and adjacent matrix
            match_matrix = feat_sim_matrix + (0.1 * adja_matrix + eps)

            # Reshape the match matrix to square matrix
            square_cost_matrix = np.zeros([max_num, max_num], np.float)

            for i in range(feat_sim_matrix.shape[0]):
                for j in range(feat_sim_matrix.shape[1]):
                    square_cost_matrix[i, j] = match_matrix[i, j]

            # Record the padding row and col index
            useless_row = []
            useless_col = []

            for i in range(feat_sim_matrix.shape[0], max_num):
                useless_row.append(i)
            for j in range(feat_sim_matrix.shape[1], max_num):
                useless_col.append(j)

            # Get the match index result
            row_ind, col_ind = test_utils.hungary(square_cost_matrix)

            ori_his_len = len(his_duration_list)
            his_matched_matrix = np.zeros([ori_his_len], np.int)

            # Iter all match pairs, If the similarity match the condition, then allocate pairs into same track seq
            for row_ind_idx, row_item in enumerate(row_ind):
                cur_idx = row_item
                his_idx = col_ind[row_ind_idx]

                # The match idx falls in padding row indices, that means the instance are bogus instance
                if cur_idx in useless_row:
                    continue

                # Match valid
                if his_idx not in useless_col:

                    # Calculating pairs' feature similarity, adjacent relation, iou
                    matched_feat_sim = feat_sim_matrix[cur_idx, his_idx]
                    matched_adja = adja_matrix[cur_idx, his_idx]
                    matched_iou = iou_matrix[cur_idx, his_idx]

                    # Only if the feature similarity meet the threshold or paris are adjacent and meet the loc feature
                    # similarity, can they be valid pairs
                    if (matched_feat_sim >= feat_sim_thresh) or (matched_feat_sim >= feat_sim_with_loc_thresh and
                                                                 matched_adja >= 1):
                        matched_text_id = his_textid_list[his_idx]

                        # Update matched track
                        test_utils.update_track(matched_text_id, video, track_res_dict, frame_id,
                                                instance_infos[cur_idx]['ann']['bbox'], texts[cur_idx],
                                                scores[cur_idx].item())

                        # Update history data
                        his_feat_array[his_idx, :] = track_feature[cur_idx, :]
                        his_duration_list[his_idx] = 0
                        his_loc_list[his_idx] = cur_frame_pos[cur_idx]
                        his_matched_matrix[his_idx] = 1

                    # Match invalid. create new track
                    else:

                        # Append a new track in history data
                        his_textid_list.append(text_id)
                        his_duration_list.append(0)
                        his_loc_list.append(cur_frame_pos[cur_idx])
                        his_feat_array = np.row_stack((his_feat_array, np.array([track_feature[cur_idx, :]])))

                        # Append a new track
                        test_utils.update_track(text_id, video, track_res_dict, frame_id,
                                                instance_infos[cur_idx]['ann']['bbox'], texts[cur_idx],
                                                scores[cur_idx].item())

                        text_id = text_id + 1

                # The match idx falls in padding col indices, instance do not match any history data, create new track
                else:

                    # Append a new track in history data
                    his_textid_list.append(text_id)
                    his_duration_list.append(0)
                    his_loc_list.append(cur_frame_pos[cur_idx])
                    his_feat_array = np.row_stack((his_feat_array, np.array([track_feature[cur_idx, :]])))

                    # Append a new track
                    test_utils.update_track(text_id, video, track_res_dict, frame_id,
                                            instance_infos[cur_idx]['ann']['bbox'], texts[cur_idx],
                                            scores[cur_idx].item())

                    text_id = text_id + 1

            # Updating history data
            his_textid_list, his_duration_list, his_loc_list, his_feat_array = test_utils. \
                update_history(his_textid_list, his_duration_list, his_loc_list, his_feat_array, his_matched_matrix,
                               max_exist_duration, ori_his_len)

    # Output
    out_file_name = os.path.join(out_dir, cfg.out_file)
    with open(out_file_name, 'w') as write_file:
        json.dump(track_res_dict, write_file, indent=4)
