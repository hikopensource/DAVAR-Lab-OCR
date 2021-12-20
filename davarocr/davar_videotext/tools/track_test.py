"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    track_test.py
# Abstract       :    generate track result from detection result

# Current Version:    1.0.0
# Date           :    2021-06-01
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
from mmdet.datasets.pipelines import Compose
from sklearn.metrics.pairwise import cosine_similarity

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

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.data.test.test_mode = True
    test_pipeline = Compose(cfg.test_pipeline)
    model_path = cfg.ckpts[0]['ModelPath']
    if not os.path.exists(model_path):
        print(model_path + ' not exist.')

    config_cfg = mmcv.Config.fromfile(cfg.ckpts[0]['ConfigPath'])
    test_cfg = config_cfg.test_cfg

    # build recognition model
    model = build_recognizor(config_cfg.model, train_cfg=None, test_cfg=test_cfg)

    # load the model pth file
    checkpoint = load_checkpoint(model, model_path, map_location='cpu')

    model.CLASSES = None

    model = MMDataParallel(model, device_ids=[0])
    device = next(model.parameters()).device
    model.eval()

    # config param
    feat_sim_thresh = 0.9
    feat_sim_withloc_thresh = 0.85
    max_exist_duration = 8
    feat_channels = 256
    eps = 1e-7
    video_list = []

    # main tracking process

    text_id = 0  # the only identification of text sequence in a video
    ori_det_data = mmcv.load(cfg.testsets[0]["AnnFile"])  # the predict detection result file by detection model
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

        # history data to match current data
        his_feat_array = []
        his_textid_list = []
        his_duration_list = []
        his_loc_list = []
        his_img_name_list = []

        # to save track sequence for specific video
        track_res_dict[video] = dict()

        # the frame id in video should start from "1" and should be consecutive by default
        for frame_id in range(1, frame_nums + 1):
            print("processing frame :", frame_id)
            key = video + '/' + str(frame_id) + '.jpg'
            img_info = value[key]

            # read predict bboxes from one image into list
            instance_infos = test_utils.instance_to_list(img_info, key)

            cur_frame_pos = []
            track_feature = None

            # data pipelines and model output
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

            # begin to record and match history data

            # In the begging, when no history data to match, create all bboxes as new track sequences
            if len(his_textid_list) == 0:
                if len(instance_infos) > 0:
                    first_img_flag = True
                    for text_img_idx, instance in enumerate(instance_infos):
                        his_textid_list.append(text_id)
                        his_img_name_list.append(instance)
                        his_duration_list.append(0)
                        his_loc_list.append(instance['ann']['bbox'])
                        if first_img_flag:
                            his_feat_array = np.array(track_feature[text_img_idx, :])
                            first_img_flag = False
                        else:
                            his_feat_array = np.row_stack((his_feat_array, np.array([track_feature[text_img_idx, :]])))

                        if text_id not in track_res_dict[video].keys():
                            track_res_dict[video][text_id] = dict()
                            track_res_dict[video][text_id]['track'] = list()
                            track_res_dict[video][text_id]['trackID'] = list()
                            track_res_dict[video][text_id]['scores'] = list()

                        key_info = str(frame_id) + ','
                        bbox = instance['ann']['bbox']
                        for point in bbox:
                            key_info += str(point) + '_'
                        key_info = key_info.strip('_')
                        key_info += ',' + texts[text_img_idx]

                        track_res_dict[video][text_id]['track'].append(key_info)
                        track_res_dict[video][text_id]['trackID'].append(instance['ann']['trackID'])
                        track_res_dict[video][text_id]['scores'].append(scores[text_img_idx].item())
                        text_id = text_id + 1
                continue

            # if current img no text
            if len(instance_infos) == 0:
                # updating duration for those unmatched
                process_num = 0
                delete_idx = 0
                ori_his_len = len(his_textid_list)
                while process_num < ori_his_len:
                    his_duration_list[delete_idx] = his_duration_list[delete_idx] + 1
                    # clear very old his text seq
                    if his_duration_list[delete_idx] >= max_exist_duration:
                        del his_textid_list[delete_idx]
                        del his_img_name_list[delete_idx]
                        del his_duration_list[delete_idx]
                        del his_loc_list[delete_idx]
                        his_feat_array = np.delete(his_feat_array, delete_idx, axis=0)
                    else:
                        delete_idx = delete_idx + 1
                    process_num = process_num + 1
                continue

            # for case only one history feature exists, should reshape to (1, feat_channels)
            if len(his_feat_array.shape) == 1 and his_feat_array.shape[0] == feat_channels:
                his_feat_array = his_feat_array.reshape(1, his_feat_array.shape[0])

            max_num = max(len(track_feature), len(his_feat_array))

            # calc feature similarty, iou, adjacent matrix to match history track, In YORO, we only use feat_sim_matrix
            # and adja_maxtrix to match, You can adjust to your only task
            feat_sim_matrix = cosine_similarity(track_feature, his_feat_array)
            iou_matrix = np.zeros([len(cur_frame_pos), len(his_loc_list)], np.float)
            adja_matrix = np.zeros([len(cur_frame_pos), len(his_loc_list)], np.int)

            for cur_idx, cur_bbox in enumerate(cur_frame_pos):
                cur_poly = test_utils.polygon_from_points(cur_bbox)

                for his_idx, his_bbox in enumerate(his_loc_list):
                    # calculate IOU
                    his_poly = test_utils.polygon_from_points(his_bbox)
                    cur_iou = test_utils.get_intersection_over_union(cur_poly, his_poly)
                    iou_matrix[cur_idx, his_idx] = cur_iou

                    # calculate adjacent matrix
                    his_poly_range_start_x = min(his_bbox[0], his_bbox[2],
                                                 his_bbox[4], his_bbox[6])
                    his_poly_range_end_x = max(his_bbox[0], his_bbox[2],
                                               his_bbox[4], his_bbox[6])
                    his_poly_range_start_y = min(his_bbox[1], his_bbox[3],
                                                 his_bbox[5], his_bbox[7])
                    his_poly_range_end_y = max(his_bbox[1], his_bbox[3],
                                               his_bbox[5], his_bbox[7])
                    # the bbox center coordinates
                    his_poly_center_x = 0.5 * (his_poly_range_start_x + his_poly_range_end_x)
                    his_poly_center_y = 0.5 * (his_poly_range_start_y + his_poly_range_end_y)

                    his_expand_range_start_x = his_poly_center_x - 1. * (his_poly_range_end_x - his_poly_range_start_x)
                    his_expand_range_end_x = his_poly_center_x + 1. * (his_poly_range_end_x - his_poly_range_start_x)
                    his_expand_range_start_y = his_poly_center_y - 1. * (his_poly_range_end_y - his_poly_range_start_y)
                    his_expand_range_end_y = his_poly_center_y + 1. * (his_poly_range_end_y - his_poly_range_start_y)
                    for i in range(4):
                        if his_expand_range_start_x <= cur_frame_pos[cur_idx][2 * i] <= his_expand_range_end_x and \
                                his_expand_range_start_y <= cur_frame_pos[cur_idx][2 * i + 1] <= \
                                his_expand_range_end_y:
                            adja_matrix[cur_idx, his_idx] = 1

            match_matrix = feat_sim_matrix + (0.1 * adja_matrix + 1e-3)

            # aggregate feature similarity matrix and adjacent matrix
            square_cost_matrix = np.zeros([max_num, max_num], np.float) + 1e-7

            for i in range(feat_sim_matrix.shape[0]):
                for j in range(feat_sim_matrix.shape[1]):
                    square_cost_matrix[i, j] = match_matrix[i, j]

            useless_row = []
            useless_col = []
            for i in range(feat_sim_matrix.shape[0], max_num):
                useless_row.append(i)
            for j in range(feat_sim_matrix.shape[1], max_num):
                useless_col.append(j)

            row_ind, col_ind = test_utils.hungary(square_cost_matrix)

            # assertion of matching valid , save res to corresponding data and updating history
            ori_his_len = len(his_duration_list)
            his_matched_matrix = np.zeros([ori_his_len], np.int)

            for row_ind_idx, row_item in enumerate(row_ind):
                cur_idx = row_item
                his_idx = col_ind[row_ind_idx]
                if cur_idx in useless_row:
                    continue

                # match valid
                if his_idx not in useless_col:
                    matched_feat_sim = feat_sim_matrix[cur_idx, his_idx]
                    matched_adja = adja_matrix[cur_idx, his_idx]
                    matched_iou = iou_matrix[cur_idx, his_idx]
                    if (matched_feat_sim >= feat_sim_thresh) or (matched_feat_sim >= feat_sim_withloc_thresh and
                                                                 matched_adja >= 1):
                        matched_text_id = his_textid_list[his_idx]
                        if matched_text_id not in track_res_dict[video].keys():
                            track_res_dict[video][matched_text_id] = dict()
                            track_res_dict[video][matched_text_id]['track'] = list()
                            track_res_dict[video][matched_text_id]['trackID'] = list()
                            track_res_dict[video][matched_text_id]['scores'] = list()

                        key_info = str(frame_id) + ','
                        bbox = instance_infos[cur_idx]['ann']['bbox']
                        for point in bbox:
                            key_info += str(point) + '_'
                        key_info = key_info.strip('_')
                        key_info += ',' + texts[cur_idx]

                        track_res_dict[video][matched_text_id]['track'].append(key_info)
                        track_res_dict[video][matched_text_id]['trackID'].append(instance_infos[cur_idx]['ann']
                                                                                 ['trackID'])
                        track_res_dict[video][matched_text_id]['scores'].append(scores[cur_idx].item())

                        his_feat_array[his_idx, :] = track_feature[cur_idx, :]
                        his_duration_list[his_idx] = 0
                        his_loc_list[his_idx] = cur_frame_pos[cur_idx]
                        his_img_name_list[his_idx] = instance_infos[cur_idx]
                        his_matched_matrix[his_idx] = 1

                    # match invalid. create new track
                    else:
                        his_textid_list.append(text_id)
                        his_img_name_list.append(instance_infos[cur_idx])
                        his_duration_list.append(0)
                        his_loc_list.append(cur_frame_pos[cur_idx])
                        his_feat_array = np.row_stack((his_feat_array, np.array([track_feature[cur_idx, :]])))

                        if text_id not in track_res_dict[video].keys():
                            track_res_dict[video][text_id] = dict()
                            track_res_dict[video][text_id]['track'] = list()
                            track_res_dict[video][text_id]['trackID'] = list()
                            track_res_dict[video][text_id]['scores'] = list()
                        else:
                            raise "ERROR: text id exists {}".format(text_id)

                        key_info = str(frame_id) + ','
                        bbox = instance_infos[cur_idx]['ann']['bbox']
                        for point in bbox:
                            key_info += str(point) + '_'
                        key_info = key_info.strip('_')
                        key_info += ',' + texts[cur_idx]
                        track_res_dict[video][text_id]['track'].append(key_info)
                        track_res_dict[video][text_id]['trackID'].append(instance_infos[cur_idx]['ann']['trackID'])
                        track_res_dict[video][text_id]['scores'].append(scores[cur_idx].item())
                        text_id = text_id + 1

                # new existed text, create new track
                else:
                    his_textid_list.append(text_id)
                    his_img_name_list.append(instance_infos[cur_idx])
                    his_duration_list.append(0)
                    his_loc_list.append(cur_frame_pos[cur_idx])
                    his_feat_array = np.row_stack((his_feat_array, np.array([track_feature[cur_idx, :]])))

                    if text_id not in track_res_dict[video].keys():
                        track_res_dict[video][text_id] = dict()
                        track_res_dict[video][text_id]['track'] = list()
                        track_res_dict[video][text_id]['trackID'] = list()
                        track_res_dict[video][text_id]['scores'] = list()
                    else:
                        raise "ERROR: text id exists {}".format(text_id)

                    key_info = str(frame_id) + ','
                    bbox = instance_infos[cur_idx]['ann']['bbox']
                    for point in bbox:
                        key_info += str(point) + '_'
                    key_info = key_info.strip('_')
                    key_info += ',' + texts[cur_idx]
                    track_res_dict[video][text_id]['track'].append(key_info)
                    track_res_dict[video][text_id]['trackID'].append(instance_infos[cur_idx]['ann']['trackID'])
                    track_res_dict[video][text_id]['scores'].append(scores[cur_idx].item())
                    text_id = text_id + 1

            # updating duration for those unmatched
            process_num = 0
            delete_idx = 0
            while process_num < ori_his_len:
                if his_matched_matrix[delete_idx] == 0:
                    his_duration_list[delete_idx] = his_duration_list[delete_idx] + 1
                    # clear very old his text seq
                    if his_duration_list[delete_idx] >= max_exist_duration:
                        del his_textid_list[delete_idx]
                        del his_img_name_list[delete_idx]
                        del his_duration_list[delete_idx]
                        del his_loc_list[delete_idx]
                        his_feat_array = np.delete(his_feat_array, delete_idx, axis=0)
                    else:
                        delete_idx = delete_idx + 1
                    process_num = process_num + 1
                else:
                    delete_idx = delete_idx + 1
                    process_num = process_num + 1

    out_file_name = os.path.join(out_dir, cfg.out_file)
    with open(out_file_name, 'w') as write_file:
        json.dump(track_res_dict, write_file, indent=4)
