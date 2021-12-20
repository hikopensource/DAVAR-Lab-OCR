"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    test_utils.py
# Abstract       :    function utils used in inference time

# Current Version:    1.0.0
# Date           :    2021-06-15
##################################################################################################
"""

import os
import json
import glob
import collections

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import Polygon as plg
import Levenshtein


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


def hungary(task_matrix):
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


def edit_dist_iou(first, second):
    """Calculate the edit distance iou between two words

    Args:
        first(str): the compared word 1
        second(str): the compared word 2

    Returns:
        float: the edit distance iou

    """
    edit_dist = Levenshtein.distance(first, second)
    inter = max(len(first), len(second)) - edit_dist
    union = len(first) + len(second) - inter
    return np.float(inter) / np.float(union)


def instance_to_list(img_info, key):
    """extract each bbox from one img detection result to construct a list

    Args:
        img_info(dict): one img detection result
        key(str): img filename

    Returns:
        list(dict): the list of all detection bboxes from one img

    """
    reslist = list()
    bboxes = img_info['content_ann']['bboxes']

    texts = img_info['content_ann'].get('texts', [None]*len(bboxes))
    labels = img_info['content_ann'].get('labels', [None]*len(bboxes))
    scores = img_info['content_ann'].get('scores', [None] * len(bboxes))
    track_id = img_info['content_ann'].get('trackID', [None] * len(bboxes))

    for i, bbox in enumerate(bboxes):
        if len(bbox) != 8:   # Filter out defective boxes and polygonal boxes
            continue
        reslist.append({
            'filename': key,
            'ann': {
                'text': texts[i],
                'bbox': bbox,
                'label': labels[i],
                'score': scores[i],
                'trackID': track_id[i]
            }
        })
    return reslist


def cal_cossim(array_a, array_b):
    """Calculate the cosine similarity between feature a and b

    Args:
        array_a(numpy array): the 2-dimensional feature a
        array_b(numpy array): the 2-dimensional feature b

    Returns:
        float: the cosine similarity between [-1, 1]

    """
    assert array_a.shape == array_b.shape, 'array_a and array_b should be same shape'
    a_norm = np.linalg.norm(array_a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(array_b, axis=1, keepdims=True)
    sim = np.dot(array_a, array_b.T) / (a_norm * b_norm)
    return sim


def gen_train_score(glimpses, img_info, refer, output):
    """ Generate gt trainset quality score according to the KMeans algorithm, which we only cluster the quality
    of "HIGH" or "MODERATE" samples, and choose the center as the template, then we calculate the samples' cosine
    similarity with template which belong to the same track

    Args:
        glimpses (Tensor): feature extract from attention cell
        img_info (dict): imgs information
        refer (str): the train set json file
        output (str): the output json file

    Returns:

    """
    assert len(glimpses) == len(img_info), 'glimpse num {} != img_info num {}'.format(len(glimpses), len(img_info))

    # Track dict:  key: track seq id, value: [[], [], []] for glimpse index, quality label, care label
    track_dict = dict()

    # Iter all samples, to get all track info
    for i, instance in enumerate(img_info):

        track_id = instance['img_info']['ann']['trackID']
        care = instance['img_info']['ann']['care']
        quality = instance['img_info']['ann']['quality']

        if track_id not in track_dict.keys():
            # index, quality, care
            track_dict[track_id] = [[], [], []]

        track_dict[track_id][0].append(i)
        track_dict[track_id][1].append(quality)
        track_dict[track_id][2].append(care)

    for key, value in track_dict.items():
        print("processing track id: ", key)
        indices = value[0]
        quality = value[1]
        care = value[2]

        # save index for high quality
        high_indices = []

        # save index for moderate quality
        mid_indices = []

        for idx, item in enumerate(indices):
            if quality[idx] == 'HIGH' and care[idx] == 1:
                high_indices.append(item)
            if quality[idx] == 'MODERATE' and care[idx] == 1:
                mid_indices.append(item)

        # KMeans first choose high quality samples,if not exists, choose moderate samples, ignore low quality
        if len(high_indices) == 0:
            high_indices = mid_indices
        if len(high_indices) == 0:
            continue

        # choose glimpse to cluster
        tmp_glimpse = []
        for index in high_indices:
            tmp_glimpse.append(glimpses[index])
        tmp_glimpse = np.stack(tmp_glimpse)
        clf = KMeans(n_clusters=1)
        clf.fit(tmp_glimpse)
        center = clf.cluster_centers_

        for idx in indices:
            sim = cal_cossim(np.expand_dims(glimpses[idx], 0), center)
            sim = max(0, sim)
            img_info[idx]['img_info']['ann']['score'] = float(sim)

    format_json(img_info, refer, output)


def gen_pred_text(pred_texts, img_info):
    """Get the predict texts to save

    Args:
        pred_texts(list): predict texts
        img_info(list): the imgs information respectively

    Returns:
        list(dict): updated img_info which add predict texts

    """
    for i, instance in enumerate(img_info):
        instance['img_info']['ann']['text'] = pred_texts[i]

    return img_info


def pred_test_score(scores, img_info, score_path, refer_file):
    """Get the predict quality score to save

    Args:
        scores(list(float)): predict texts
        img_info(list(dict)): the imgs information respectively
        score_path(str): the output json quality score file
        refer_file(str): the referred testset file in case of lack of some imgs key

    Returns:

    """

    with open(refer_file, 'r') as refer_f:
        refer_dict = json.load(refer_f)
        rcg_res = dict()

        # to format the bbox instances into img instances
        for i, instance in enumerate(img_info):
            instance['img_info']['ann']['score'] = scores[i].item()
            key = instance['img_info']['filename']
            if key not in rcg_res.keys():
                rcg_res[key] = {}
                rcg_res[key]['content_ann'] = dict()
                rcg_res[key]['content_ann']['scores'] = []
                rcg_res[key]['content_ann']['texts'] = []
                rcg_res[key]['content_ann']['bboxes'] = []
            rcg_res[key]['content_ann']['scores'].append(instance['img_info']['ann']['score'])
            rcg_res[key]['content_ann']['texts'].append(instance['img_info']['ann']['text'])
            rcg_res[key]['content_ann']['bboxes'].append(instance['img_info']['ann']['bbox'])

        # in case of some img do not exists any bbox, so the img_info will be lack of these images information, we
        # should add these information
        for key in refer_dict.keys():
            if key not in rcg_res.keys():
                rcg_res[key] = {}
                rcg_res[key]['content_ann'] = dict()
                rcg_res[key]['content_ann']['scores'] = []
                rcg_res[key]['content_ann']['texts'] = []
                rcg_res[key]['content_ann']['bboxes'] = []

    with open(score_path, "w", encoding="utf-8") as write_file:
        json.dump(rcg_res, write_file, indent=4)


def filter_punctuation(sentence, punctuation=':(\'-,%>.[?)"=_*];&+$@/|!<#`{~\}^'):
    """
    Args:
        sentence (str): string which needs to filter the punctuation
        punctuation (str): the punctuation which is unnecessary

    Returns:
        result (str): string without the unnecessary punctuation

    """
    temp_result = []
    for item in sentence:

        # ':(\'-,%>.[?)"=_*];&+$@/|!<#`{~\}^':
        if item in punctuation:
            continue
        temp_result.append(item)
    result = "".join(temp_result)
    return result


def load_json(json_file, window_size):
    """ Load json file for yoro detection inference

    Args:
        json_file (str): the path of test set
        window_size(int): the window size of consecutive frames

    Returns:
        data_infos (list(dict)): the list format of test data

    Returns:
        indices (list(list(int))): indices of consecutive frames
    """
    with open(json_file, 'r') as read_file:
        ann = json.load(read_file, object_pairs_hook=collections.OrderedDict)
    data_infos = list()
    # format the testset info
    for key in ann.keys():
        tmp_dict = dict()
        tmp_dict["filename"] = key
        tmp_dict["height"] = ann[key]["height"]
        tmp_dict["width"] = ann[key]["width"]
        tmp_dict["ann"] = ann[key]["content_ann"]
        tmp_dict["ann2"] = ann[key].get("content_ann2", None)
        tmp_dict["video"] = ann[key]["video"]
        tmp_dict["frameID"] = ann[key]["frameID"]
        data_infos.append(tmp_dict)

    # group the data by video
    video_group = {}
    for data in data_infos:
        video_name = data['video']
        if video_name not in video_group.keys():
            video_group[video_name] = [data]
        else:
            video_group[video_name].append(data)

    # sorted each video data by frame id
    sorted_img_infos = []
    for video_data in video_group.values():
        video_data = sorted(video_data, key=lambda x: int(x['frameID']))
        sorted_img_infos += video_data
    data_infos = sorted_img_infos

    # generate the test indices
    indices = generate_index(data_infos, window_size)
    return data_infos, indices


def generate_index(data_infos, window_size):
    """ Generate the indices of consecutive frames

    Args:
        data_infos (str): the data informations
        window_size(int): the window size of consecutive frames

    Returns:
        list(list(int)): the indices which each one represent the window size consecutive frames,like:
        [[1, 2, 3, 4, 5], ..., [64, 65, 66, 67, 68]]. each number in [1, 2, 3, 4, 5] represents the index of data_infos,
        which are consecutive frames
    """
    indices = []

    # The number of left adjacent frames
    left_cnt = (window_size - 1) // 2

    # The number of right adjacent frames
    right_cnt = window_size - 1 - left_cnt

    for i, img_info in enumerate(data_infos):
        video = img_info['video']

        # Anchor frame index
        index = [i]

        # Append left adjacent frames' index
        for left_index in range(1, left_cnt + 1):
            j = i - left_index
            if j < 0 or data_infos[j]['video'] != video:
                index.insert(0, i)
            else:
                index.insert(0, j)

        # Append right adjacent frames' index
        for right_index in range(1, right_cnt + 1):
            j = i + right_index
            if j >= len(data_infos) or data_infos[j]['video'] != video:
                index.append(i)
            else:
                index.append(j)

        assert len(index) == window_size
        indices.append(index)

    return indices


def load_flows(img_info, flow_path):
    """ Load flow info for yoro detection inference

    Args:
        img_info (str): one image info
        flow_path(str): the flow saved path

    Returns:
        flow (ndarray): the optical flow which shape should be [window size - 1, H , W, 2], window size - 1 because we
    only calculate the adjacent frames to anchor frame optical flows, the anchor to anchor optical flow not saved.
    """
    video = img_info['video']
    frame_id = img_info['frameID']
    data = np.load(os.path.join(flow_path, video, str(frame_id) + '.npz'))
    flow = data['arr_0']
    return flow


def make_paths(*args):
    """
    make a new directory
    Args:
        *args (str): back parameter

    Returns:

    """
    for para in args:
        if not os.path.exists(para):
            # make the new directory
            os.makedirs(para)


def txr2json(gt_file, predict_txt_dir, out_file):
    """ Reformat the detection result .txt to json

    Args:
        gt_file (str): the gt test file for get height and width info
        predict_txt_dir (str): the predict detection txt directory
        out_file (str): the output json file

    Returns:

    """
    # Original detection output format .txt
    txt_list = glob.glob(predict_txt_dir + '/*.txt')

    # Refer gt data list for height and width info
    with open(gt_file, 'r') as read_file:
        data_gt_ori = json.load(read_file)

    det_dict = {}

    # Make sure the key is made by "Video_name/frame"
    data_gt = {}
    for key, value in data_gt_ori.items():
        video = key.split('/')[-2]
        frame = key.split('/')[-1]
        key_ = video + '/' + frame
        if key_ not in data_gt.keys():
            data_gt[key_] = value
        else:
            raise KeyError("The data has repeated key {}".format(key))

    # Iter all txt
    for txt in txt_list:

        # The txt name is made by "Video_name-frame"
        txt_name = txt.split('/')[-1]
        print("processing video :", txt_name)
        video = txt_name.split('-')[0]
        frame_id = txt_name.split('-')[-1].split('.')[0]
        key = video + '/' + frame_id + '.jpg'

        with open(txt, 'r') as read_txt:
            lines = read_txt.readlines()

        # Get height and width information
        det_dict[key] = {}
        det_dict[key]['height'] = data_gt[key]['height']
        det_dict[key]['width'] = data_gt[key]['width']
        det_dict[key]['content_ann'] = {}
        det_dict[key]['content_ann']['bboxes'] = []

        # Get predict bboxes
        for line in lines:
            bbox_vec = line.split(',')
            bbox = []
            for point in bbox_vec:
                bbox.append(int(point))
            det_dict[key]['content_ann']['bboxes'].append(bbox)

    # Check if all the frames have been saved
    for key in data_gt:
        if key not in det_dict.keys():
            raise KeyError("the testset key {} do not exists in predict result".format(key))

    # Output
    with open(out_file, 'w') as write_file:
        json.dump(det_dict, write_file, indent=4)


def format_json(data, refer, output):
    """ Reformat the train gt quality score result

    Args:
        data (str): the gt quality score results
        refer (str): the refer data list
        output (str): the output json file

    Returns:

    """
    with open(refer, 'r') as read_file:
        refer_data = json.load(read_file)

    with open(output, 'w') as write_file:

        # Update 'score' field in refer dict data
        for key in refer_data.keys():
            refer_data[key]['content_ann']['score'] = [None for i in
                                                       range(len(refer_data[key]['content_ann']['texts']))]

        # Format the data(list(dict)) to dict
        for item in data:
            img_info = item['img_info']
            filename = img_info['filename']

            # Only to deal with those samples which have quality scores
            if 'score' in img_info['ann'].keys():

                # Get this text instance track id and to match with track id in gt data
                track_id = img_info['ann']['trackID']
                track_ids = refer_data[filename]['content_ann']['trackID']

                # The flag is to record whether there exists more than one text instances that have same track id in
                # one frame
                flag = False
                for idx, i_id in enumerate(track_ids):
                    if i_id == track_id:
                        if flag:
                            print(refer_data[filename], "this frame exists two same track id instance")
                        flag = True
                        refer_data[filename]['content_ann']['score'][idx] = img_info['ann']['score']
                        refer_data[filename]['content_ann']['cares'][idx] = 1
                        if refer_data[filename]['content_ann']['texts'][idx] == '###':
                            refer_data[filename]['content_ann']['texts'][idx] = '555'

        # Output
        json.dump(refer_data, write_file, indent=4)


def update_history(his_textid_list, his_duration_list, his_loc_list, his_feat_array, his_matched_matrix,
                   max_exist_duration, ori_his_len):
    """ Reformat the train gt quality score result

    Args:
        his_textid_list (list(int)): the history data of text id
        his_duration_list (str): the history data of text id duration
        his_loc_list (str): the history data of text bbox
        his_feat_array (str): the history data of text feature
        his_matched_matrix (str): the history data of match
        max_exist_duration (str): the max exist duration time

    Returns:
        list(int): updated the history data of text id

    Returns:
        list(int): updated the history data of text id duration

    Returns:
        list(list(int)): updated the history data of text bbox

    Returns:
        numpy array: updated the history data of text feature

    """

    # Updating history data
    process_num = 0
    delete_idx = 0
    while process_num < ori_his_len:
        # If not match, updating the duration time
        if his_matched_matrix[delete_idx] == 0:
            his_duration_list[delete_idx] = his_duration_list[delete_idx] + 1

            # Clear very old his text seq
            if his_duration_list[delete_idx] >= max_exist_duration:
                del his_textid_list[delete_idx]
                del his_duration_list[delete_idx]
                del his_loc_list[delete_idx]
                his_feat_array = np.delete(his_feat_array, delete_idx, axis=0)
            else:
                delete_idx = delete_idx + 1
            process_num = process_num + 1
        else:
            delete_idx = delete_idx + 1
            process_num = process_num + 1
            
    return his_textid_list, his_duration_list, his_loc_list, his_feat_array


def calculate_expand(bbox):
    """ Reformat the train gt quality score result

    Args:
        bbox (list(int)): bbox: [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        int: expand start x

    Returns:
        int: expand end x

    Returns:
        int: expand start y

    Returns:
        int: expand end y

    """
    # Calculate adjacent matrix
    poly_start_x = min(bbox[0], bbox[2], bbox[4], bbox[6])
    poly_end_x = max(bbox[0], bbox[2], bbox[4], bbox[6])
    poly_start_y = min(bbox[1], bbox[3], bbox[5], bbox[7])
    poly_end_y = max(bbox[1], bbox[3], bbox[5], bbox[7])

    # The bbox center coordinates
    poly_center_x = 0.5 * (poly_start_x + poly_end_x)
    poly_center_y = 0.5 * (poly_start_y + poly_end_y)

    # Calculating the expand start x, y and end x, y coordinates
    expand_start_x = poly_center_x - 1. * (poly_end_x - poly_start_x)
    expand_end_x = poly_center_x + 1. * (poly_end_x - poly_start_x)
    expand_start_y = poly_center_y - 1. * (poly_end_y - poly_start_y)
    expand_end_y = poly_center_y + 1. * (poly_end_y - poly_start_y)

    return expand_start_x, expand_end_x, expand_start_y, expand_end_y


def update_track(text_id, video, track_res_dict, frame_id, bbox, text, score):
    """ Updating a track or create a new track

    Args:
        text_id (int): text sequence id
        video (str): video name
        track_res_dict (dict): dict to save each video's track seqs
        frame_id (int): frame id number
        bbox (list(int)): bbox: [x1, y1, x2, y2, x3, y3, x4, y4]
        text (str): recognition result
        score (float): quality score

    Returns:

    """
    if text_id not in track_res_dict[video].keys():
        track_res_dict[video][text_id] = dict()
        track_res_dict[video][text_id]['track'] = list()
        track_res_dict[video][text_id]['scores'] = list()

    key_info = str(frame_id) + ','
    for point in bbox:
        key_info += str(point) + '_'
    key_info = key_info.strip('_')
    key_info += ',' + text

    track_res_dict[video][text_id]['track'].append(key_info)
    track_res_dict[video][text_id]['scores'].append(score)
