import mmcv
from davarocr.davar_videotext.tools import test_utils
from davarocr.davar_common.apis import init_model, inference_model
import cv2
import time
import json
import os
import collections
import numpy as np
import torch

if __name__ == '__main__':
    # model config file
    config_file = './config/yoro_east.py'
    # checkpoint
    checkpoint_file = "/path/to/IC15_STDet_TiV-edd8dd35.pth"
    # test data list
    test_dataset='/path/to/ic13_video_test_datalist.json'
    # img prefix
    img_prefix = '/path/to/Images/'
    # the flow directory path
    flow_path = '/path/to/test_flow/'
    # path to save final prediction in .txt format
    out_put_dir = "/path/to/result/"
    # path to save visualization result.
    vis_dir = "/path/to/vis_result/"

    # file to save final prediction in json format
    result_json = "/path/to/predict_detection.json"

    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    model = init_model(config_file, checkpoint_file, device='cuda:0')
    time_sum = 0.0

    # window size is the number of consecutive frames when training, This should stay the same with train stage and
    # should be odd number
    windowSize = 5
    data_infos, indices = test_utils.load_json(test_dataset, windowSize)
    video = ''
    time_start = time.time()

    pre_features = None
    img_metas = None
    pre_video = None

    # In YORO inferenece stage, we once input window size consecutive frames in a video, then we only output the center
    # frame result. The inference order will be in the order of video frame id in a video. once all the frames in a
    # video has been tested, the test the next video

    for index in indices:

        # we only output the center one result
        target_idx = (windowSize - 1) // 2

        print("processing img: " + data_infos[index[target_idx]]['filename'])

        video = data_infos[index[target_idx]]['video']
        filename = data_infos[index[target_idx]]['filename']
        framid = int(data_infos[index[target_idx]]['frameID'])
        imgs = []

        # index represents the window size consecutive frames
        for i in index:
            img_info = {}
            img_path = img_prefix + data_infos[i]['filename']
            img_info['filename'] = img_path
            img_info['video'] = data_infos[i]['video']
            img_info['frameID'] = data_infos[i]['frameID']
            img_info['flow'] = None
            img_info['pre_features'] = None
            imgs.append(img_info)

        # only need load the center frame's optical flow
        flow_info = test_utils.load_flows(imgs[target_idx], flow_path)

        # then we save the flow and previous features in the last frame
        imgs[-1]['flow'] = flow_info
        imgs[-1]['pre_features'] = pre_features

        # this means that we now deal with a new video, we should init the previous features and video
        if pre_video != video:
            imgs[-1]['pre_features'] = None
            pre_video = video
            final_result = inference_model(model, imgs)
            pre_features = final_result['pre_features']
            img_metas = final_result['img_metas']
        else:
            final_result = inference_model(model, [imgs[-1]])
            pre_features = pre_features[1:]
            img_metas = img_metas[1:]
            pre_features = torch.cat((pre_features, final_result['pre_features']), dim=0)
            img_metas.extend(final_result['img_metas'])

        img = mmcv.imread(imgs[target_idx]['filename'])
        img_copy = img.copy()

        bboxes = []
        save_txt_name = filename.split("/")[-2] + '-' + filename.split("/")[-1].split(".")[0]
        txt = open(out_put_dir + "{}.txt".format(save_txt_name), "w")
        result = final_result['bboxes']

        for img_id in range(len(result)):
            for i in range(len(result[img_id]["points"])):
                points2 = result[img_id]["points"][i].tolist()
                for j in range(0, len(points2), 2):
                    points2[j] = int(points2[j])
                    points2[j + 1] = int(points2[j + 1])
                    points2[(j + 2) % len(points2)] = int(points2[(j + 2) % len(points2)])
                    points2[(j + 3) % len(points2)] = int(points2[(j + 3) % len(points2)])
                    cv2.circle(img_copy, (points2[j], points2[(j+1)]), 5, (0, 255, 255),-1)
                    cv2.line(img_copy, (points2[j], points2[j+1]), (points2[(j+2)%len(points2)], points2[(j+3)%len(points2)]),
                             (0, 0, 255), 2)
                    txt.write("{},{}".format(points2[j], points2[j+1]))
                    if j != len(points2) - 2:
                        txt.write(",")
                    elif i != len(result[img_id]["points"])-1:
                        txt.write("\n")
                points = list(map(int, points2))
                bboxes.append(points)
            txt.close()

        if not os.path.exists(vis_dir + video):
            os.makedirs(vis_dir + video)

        cv2.imwrite(vis_dir + video + '/' + imgs[target_idx]['filename'].split('/')[-1], img_copy)

    test_utils.txr2json(test_dataset, out_put_dir, result_json)
    time_end = time.time()
    print('total time: {}'.format(time_end - time_start))
