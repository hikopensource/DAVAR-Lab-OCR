"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    chargrid_data.py
# Abstract       :    Chargrid data preprocessing including generation and formation

# Current Version:    1.0.0
# Date           :    2022-03-23
##################################################################################################
"""

import os
import string
import cv2
import mmcv
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt

from string import punctuation
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from davarocr.davar_layout.datasets.pipelines import MMLAFormatBundle
from mmdet.datasets.pipelines.formating import to_tensor


@PIPELINES.register_module()
class ChargridDataGeneration():
    """
    Generating chargrid map & corresponding segmentation
    map from gt_bboxes, texts & labels.

    Args:
        vocab (str): vocab.txt, None to use default character set
        visualize(bool): whether to visualize chargrid map
        vis_save_dir(str): path to save visualization result
        use_tokenization (bool): whether to use tokenization
        tokenizer (str): tokenizer type, e.g. BertWordPieceTokenizer
        with_label (bool): whether to output chargrid_label,
            False in test process
        poly_shape (bool): whether to fill the chargrid in poly format
            (or box format)
    """

    def __init__(self,
                 vocab=None,
                 visualize=False,
                 vis_save_dir=None,
                 use_tokenization=False,
                 tokenizer=None,
                 with_label=True,
                 poly_shape=False,
                 with_bieo_label=False,
                 ):
        self.vocab = vocab
        self.visualize = visualize
        self.use_tokenization = use_tokenization
        self.tokenizer = tokenizer
        self.with_label = with_label
        self.poly_shape = poly_shape
        self.with_bieo_label = with_bieo_label

        if visualize:
            if vis_save_dir is not None:
                self.vis_save_dir = osp.abspath(vis_save_dir)
                mmcv.mkdir_or_exist(osp.abspath(vis_save_dir))
            else:
                raise ValueError("Visualization result path should be specified!")

        if vocab is not None:
            if os.path.exists(self.vocab):
                if self.use_tokenization:
                    # tokenization by tokenizer, e.g. BertWordPieceTokenizer
                    pass
                else:
                    # tokenization by vocab file.
                    with open(self.vocab, 'r', encoding='utf8') as read_f:
                        all_words = read_f.readline().strip()
                    default_token = ['[BG]', '[UNK]']
                    all_words = list(all_words)
                    self.character = default_token + all_words

                    # default 0 to background
                    self.word2idx = {char: idx for idx, char in enumerate(self.character)}
            else:
                raise ValueError("vocab.txt is required in chargrid datageneration...")
        else:
            all_words = list(string.ascii_letters + string.digits + punctuation)
            default_token = ['[BG]', '[UNK]']
            self.character = default_token + all_words

            # default 0 to background
            self.word2idx = {char: idx for idx, char in enumerate(self.character)}

    def __call__(self, results):
        img = results['img']

        gt_labels = []
        if self.with_label:
            gt_labels = results['gt_labels'] if results.get('gt_labels', None) is not None else results['gt_labels_2']
        if self.with_bieo_label:
            gt_bieo_labels = results['gt_bieo_labels']
        gt_bboxes = results['gt_bboxes'] if results.get('gt_bboxes', None) is not None else results['gt_bboxes_2']
        gt_texts = results['gt_texts']

        h_img, w_img = img.shape[0], img.shape[1]
        chargrid_data = np.zeros([h_img, w_img], dtype=np.int64)
        chargrid_label = np.zeros([h_img, w_img], dtype=np.int64)

        for line_idx, per_bbox in enumerate(gt_bboxes):
            if self.with_label:
                # default 0 to bg class
                per_label = gt_labels[line_idx] + 1
            if self.with_bieo_label:
                per_bieo_label = gt_bieo_labels[line_idx]
            per_text = gt_texts[line_idx]
            real_content = per_text.strip()

            per_box_tmp = per_bbox.reshape((-1, 2))
            start_w, start_h, end_w, end_h = np.min(per_box_tmp[:, 0]), \
                                             np.min(per_box_tmp[:, 1]), \
                                             np.max(per_box_tmp[:, 0]), \
                                             np.max(per_box_tmp[:, 1])
            start_w = int(start_w)
            start_h = int(start_h)
            end_w = int(end_w)
            end_h = int(end_h)

            fake_data = np.zeros_like(chargrid_data)
            fake_label = np.zeros_like(chargrid_data)
            fake_mask = np.zeros_like(chargrid_data).astype(np.uint8)

            if real_content:
                char_span = int((end_w - start_w) / len(real_content))
                if self.use_tokenization and self.tokenizer:
                    pass
                else:
                    for char_idx, per_char in enumerate(real_content):
                        if per_char in self.word2idx.keys():
                            fake_data[start_h:end_h,
                            start_w + char_idx * char_span: start_w + (char_idx + 1) * char_span] = self.word2idx[
                                per_char]
                        else:
                            fake_data[start_h:end_h,
                            start_w + char_idx * char_span: start_w + (char_idx + 1) * char_span] = self.word2idx[
                                '[UNK]']

                        if self.with_label:
                            fake_label[start_h:end_h,
                            start_w + char_idx * char_span: start_w + (char_idx + 1) * char_span] = int(per_label)
                        if self.with_bieo_label:
                            fake_label[start_h:end_h,
                            start_w + char_idx * char_span: start_w + (char_idx + 1) * char_span] = per_bieo_label[
                                char_idx]

            if self.poly_shape:
                poly = np.array(per_bbox).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(fake_mask, [poly], 1)
            else:
                poly = np.array([[start_w, start_h], [end_w, start_h],
                                 [end_w, end_h], [start_w, end_h]]).astype(np.int32)
                cv2.fillPoly(fake_mask, [poly], 1)

            fake_mask = fake_mask.astype(np.int64)
            fake_data = fake_data * fake_mask
            fake_label = fake_label * fake_mask

            chargrid_data = chargrid_data * (1 - fake_mask) + fake_data
            if self.with_label or self.with_bieo_label:
                chargrid_label = chargrid_label * (1 - fake_mask) + fake_label

        results['chargrid_map'] = np.eye(len(self.character))[chargrid_data]
        results['chargrid_masks'] = chargrid_label[np.newaxis, :, :]

        #################################draw img##############################
        # rgb_cad = [255, 0, 128, 80, 20, 196, 0, 100, 200, 150, 255, 60]
        # chargrid_r = chargrid_data.copy()
        # chargrid_g = chargrid_data.copy()
        # chargrid_b = chargrid_data.copy()
        # for i in range(len(chargrid_data)):
        #     for j in range(len(chargrid_data[i])):
        #         chargrid_r[i][j] = chargrid_data[i][j] * rgb_cad[chargrid_data[i][j] % len(rgb_cad)]
        #         chargrid_g[i][j] = chargrid_data[i][j] * rgb_cad[chargrid_data[i][j] % len(rgb_cad) - 1]
        #         chargrid_b[i][j] = chargrid_data[i][j] * rgb_cad[chargrid_data[i][j] % len(rgb_cad) - 2]
        # chargrid_img = np.array([chargrid_r, chargrid_g, chargrid_b])
        # chargrid_img = chargrid_img.swapaxes(0, 1).swapaxes(1, 2)

        # visualize chargrid map
        if self.visualize:
            # Rearrange chargrid
            chargrid_plt = chargrid_data.astype(np.float64)
            # Relocate class 1 '[UNK]' to the end
            chargrid_plt[chargrid_plt == 1] = len(self.character)
            chargrid_plt[chargrid_plt > 0] -= 1
            # Normalize
            chargrid_plt[...] = chargrid_plt[...] / (len(self.character) - 1) * 255
            chargrid_plt = chargrid_plt.astype(np.uint8)
            # Get RGB chargrid map
            chargrid_img = cv2.applyColorMap(chargrid_plt, cv2.COLORMAP_JET)
            # print(chargrid_img.shape)
            # raise KeyboardInterrupt
            for i in range(chargrid_img.shape[2]):
                chargrid_img[..., i][chargrid_plt == 0] = 0
                chargrid_img[..., i][chargrid_plt == 255] = 255
            # cmap = plt.get_cmap('gnuplot2')
            # chargrid_rgba = cmap(chargrid_plt)
            # chargrid_rgba[...] *= 255
            # chargrid_rgba = chargrid_rgba.astype(np.uint8)
            # To BGR
            # chargrid_bgr = cv2.cvtColor(chargrid_rgba, cv2.COLOR_RGBA2BGR)

            # save result
            img_name = results['filename'].split('/')[-1]
            cv2.imwrite(osp.join(self.vis_save_dir, img_name), chargrid_img)
        return results

    def __repr__(self):
        return self.__class__.__name__ + \
               '(vocab={}, use_tokenization={}, tokenizer={})'.format(
                   self.vocab, self.use_tokenization, self.tokenizer)


@PIPELINES.register_module()
class ChargridFormatBundle(MMLAFormatBundle):
    """
    Format Chargrid data

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - chargrid map: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - chargrid label: (1)to tensor, (2)to DataContainer (stack=True)
    """

    def __call__(self, results):
        super(ChargridFormatBundle, self).__call__(results)
        if 'chargrid_map' in results:
            chargrid_map = np.ascontiguousarray(
                results['chargrid_map'].transpose(2, 0, 1))
            results['chargrid_map'] = DC(
                to_tensor(chargrid_map).float(), stack=True)
        if 'chargrid_masks' in results:
            results['chargrid_masks'] = DC(
                to_tensor(results['chargrid_masks']), stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__
