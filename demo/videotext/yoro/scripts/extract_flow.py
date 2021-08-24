# -*- coding: utf-8 -*-
"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    optical_flow_extract.py
# Abstract       :    extract optical flow script

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
import cv2
import numpy as np
import glob
import os


if __name__ == '__main__':

    # path where to read the images
    path = '/path/to/Images'

    # output path where to save the optical flow
    output_path = '/path/to/output'

    videos = glob.glob(path + '/*')

    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    # this should stay same with train and test stage
    window_size = 5
    for video in videos:
        video_name = video.split('/')[-1]
        print("processing video: " + str(video_name))
        # the image's format is .jpg or .JPG by default
        frames = list(glob.glob(video + '/*.jpg')) + list(glob.glob(video + '/*.JPG'))
        # sort the frames by frame id
        frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('.')[0]))

        # the center frame left and right adjacent number
        adjacent_num = (window_size - 1) // 2
        for i in range(len(frames)):
            print("processing frames: " + frames[i])
            index = []
            for left in range(i - adjacent_num, i):
                # if left adjacent number is not enough, we add the first frame
                left = max(left, 0)
                index.append(left)
            for right in range(i + 1, i + adjacent_num + 1):
                # if right adjacent number is not enough, we add the last frame
                right = min(len(frames) - 1, right)
                index.append(right)

            video = frames[i].split('/')[-2]
            frameID = (frames[i].split('/')[-1].split('.')[0])
            if not os.path.exists(os.path.join(output_path, '{}/'.format(video))):
                os.makedirs(os.path.join(output_path, '{}/'.format(video)))

            # for saving storage, we resize the image to (240, 320) to extract flow
            cur = cv2.imread(frames[i])
            cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            cur = cv2.resize(cur, (320, 240), interpolation=cv2.INTER_AREA)

            # extract optical flow between frame(t + i) and frame(t)
            flows = []
            for j in index:
                prev = cv2.imread(frames[j])
                prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
                prev = cv2.resize(prev, (320, 240), interpolation=cv2.INTER_AREA)
                flow = TVL1.calc(cur, prev, None)
                flows.append(flow)

            # saving optical flow
            flows = np.stack(flows)
            np.savez(os.path.join(output_path, '{}/{}.npz'.format(video, int(frameID))), flows)
