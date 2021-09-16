"""
#################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    show_masks.py
# Abstract       :    Script for visualization

# Current Version:    1.0.0
# Date           :    2021-06-24
#################################################################################################
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import time
import imageio
import json
import torch
import pyclipper
import mmcv

colorlist=[(255,0,0), (0, 255, 0), (0,0, 255), (255,255,0), (255,0,255), (0,255,255)]

def add_text(img, text, left, top, textColor, textSize):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("./simsun.ttc", textSize, encoding="utf-8")
    draw.text((max(left,0), max(top,0)), text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def show_segmentation(img, mask_result, bboxes_result=None, alpha=0.5, num_grid=40, out_prefix="./"):

    img_mask = np.zeros(img.shape, np.uint8)
    img_mask[:, :, 2] = (mask_result * 255).astype(np.uint8)
    img_mask = cv2.addWeighted(img_mask, alpha, img, 1 - alpha, 0)

    H, W, _ = img.shape
    step_h = H / num_grid
    step_w = W / num_grid

    for i in range(num_grid):
        cv2.line(img_mask, (0, int(i * step_h)), (W, int(i * step_h)), (0, 0, 0))
        img_mask = add_text(img_mask, str(i), 0, int(i * step_h), (255, 255, 255),
                            textSize=int(min(step_h, step_w)))
        cv2.line(img_mask, (int(i * step_w), 0), (int(i * step_w), H), (0, 0, 0))
        img_mask = add_text(img_mask, str(i), int(i * step_w), 0, (255, 255, 255),
                            textSize=int(min(step_h, step_w)))
    cv2.imwrite(out_prefix + '_seg.jpg', img_mask)


def show_mask_att(img, mask_att_results, alpha=0.5, out_prefix="./"):
    img_mask_att = []

    for c_id in range(10):
        mask_att_c = mask_att_results[c_id, :, :]
        mask_att_c = mask_att_c - np.min(mask_att_c)
        mask_att_c = mask_att_c / np.max(mask_att_c)
        heatmap = cv2.applyColorMap(np.uint8(255 * mask_att_c), cv2.COLORMAP_JET)
        cam = (heatmap + np.float32(img)) / 255
        cam = cam / np.max(cam)
        img_mask_att_c = np.uint8(255 * cam)
        img_mask_att_c = cv2.putText(img_mask_att_c, 'Frame:' + str(c_id), (0, img.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 0, 0), 2)
        img_mask_att.append(img_mask_att_c[:,:,::-1])
    imageio.mimsave(out_prefix + '_cma.gif', img_mask_att, 'GIF', duration=1)


def show_text(img, texts_result, bboxes_result, out_prefix="./"):
    img_draw = img.copy()
    for j, box in enumerate(bboxes_result):
        for i in range(0, len(box), 2):
            cv2.line(img_draw, (int(box[i]), int(box[i+1])),(int(box[(i+2)%len(box)]),  int(box[(i+3)%len(box)])), (255,0,0),2)
        img_draw = add_text(img_draw, str(texts_result[j]), int(box[i]-5), int(box[i+1]-5), (0, 0, 0), textSize=15)
    cv2.imwrite(out_prefix + '_text.jpg', img_draw)

def show_cate(img, cate_preds, resize_shape=(1000,1000), pad_size_divisor=128, grid_num=40, out_prefix="./"):
    def PadResize(img, resize_shape=(1000, 1000), size_divisor=128):
        img = mmcv.imrescale(img, resize_shape)
        padded_img = mmcv.impad_to_multiple(img, size_divisor)
        return padded_img

    img_cate = img.copy()
    img_cate = PadResize(img_cate, resize_shape=resize_shape, size_divisor=pad_size_divisor )
    H, W, _ = img_cate.shape
    step_h = H / grid_num
    step_w = W / grid_num

    for i in range(grid_num):
        cv2.line(img_cate, (0, int(i * step_h)), (W, int(i * step_h)), (0, 0, 0))
        cv2.line(img_cate, (int(i * step_w), 0), (int(i * step_w), H), (0, 0, 0))
        for j in range(grid_num):
            if cate_preds[i*grid_num + j] > 0:
                cv2.rectangle(img_cate, (int(j*step_w),int(i * step_h)), (int((j+1)*step_w), int((i+1) * step_h)), colorlist[cate_preds[i*grid_num + j]%6],-1)

    cv2.imwrite(out_prefix + '_cate.jpg', img_cate)
