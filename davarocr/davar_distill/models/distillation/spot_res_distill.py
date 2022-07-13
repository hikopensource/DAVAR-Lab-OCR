"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    spot_res_distill.py
# Abstract       :    The main process of text spotting resolution distillation

# Current Version:    1.0.0
# Date           :    2022-07-07
##################################################################################################
"""
import math
import mmcv
import numpy as np

import torch
import torch.nn.functional as F

from mmdet.core import BitmapMasks
from mmcv.runner.checkpoint import load_checkpoint
from davarocr.davar_common.models import build_connect
from davarocr.davar_spotting.models import SPOTTER
from davarocr.davar_spotting.models.builder import build_spotter

from .base import BaseDistillation
from ...core import beam_decode


@SPOTTER.register_module()
class SpotResolutionDistillation(BaseDistillation):
    """ The main process of text spotting resolution distillation """

    def __init__(self,
                 student,
                 teacher,
                 policy=None,
                 kd_cfg=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        """
        Args:
            student (dict): student network architecture
            teacher (dict): teacher network architecture
            policy (dict): resolution selector policy parameter
            kd_cfg (dict): knowledge distillation cfg parameter
            train_cfg (mmcv.Config): model training cfg parameter
            test_cfg (mmcv.Config): model training cfg parameter
            pretrained (str, optional): model path of the pre_trained model
        """
        super().__init__()
        self.stu_cfg = student
        self.tea_cfg = teacher
        self.kd_cfg = kd_cfg
        self.policy_cfg = policy

        self.stu = build_spotter(self.stu_cfg)
        self.tea = build_spotter(self.tea_cfg)
        self.policy = build_connect(self.policy_cfg) if self.policy_cfg is not None else None

        # Load checkpoint
        if self.stu_cfg.pretrained is not None:
            load_checkpoint(self.stu, self.stu_cfg.pretrained, strict=False)

        if self.tea_cfg.pretrained is not None:
            load_checkpoint(self.tea, self.tea_cfg.pretrained, strict=False)

        for param in self.tea.parameters():
            param.requires_grad = False

    def forward_train(self,
                      img,
                      img_metas,
                      runner,
                      **kwargs):
        """ Forward train process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            runner (mmcv.runner): mmcv runner
            **kwargs: other parameters

        Returns:
            dict: all losses in a dict
        """
        losses = dict()

        # Teacher network forward procedure
        with torch.no_grad():
            tea_output = self.tea.forward_train(img, img_metas, is_train=False, **kwargs)

        tea_rcg_pred = tea_output['rcg_pred']
        tea_roi_feat = tea_output['rcg_roi_feat']
        tea_context_feat = tea_output['rcg_context_feat']
        recog_target = tea_output['recog_target']
        gt_texts = kwargs['gt_texts']

        # Resolution selector training procedure
        if self.policy is not None:
            temperature = self.policy_cfg.temperature * \
                          (self.policy_cfg.temperature_decay ** runner._epoch)
            # Calculate the selected resolution
            decisions = []
            for i in range(img.size(0)):
                height, width, _ = img_metas[i]['pad_shape']
                ori_img = img[i][:, :height, : width].unsqueeze(0)
                result = self.policy(ori_img)
                result = F.gumbel_softmax(result, tau=temperature, hard=True)
                decisions.append(result)
            decisions = torch.cat(decisions)
            # Initialization
            rcg_decisions = []
            rs_rcg_pred = []
            res_num = len(self.policy_cfg.scale_factor)
            penalty = torch.zeros((img.size(0), res_num), dtype=decisions.dtype, device=decisions.device)
            # Iterate over images
            for batch_id in range(img.size(0)):
                batch_rcg_pred = []
                # Iterate over different resolutions
                for res_id in range(res_num):
                    single_img = img[batch_id].unsqueeze(0)
                    single_img_metas = [img_metas[batch_id]]
                    # Calculate the scaled resolution based on the short edge
                    single_res = [int(self.policy_cfg.scale_factor[res_id] *
                                      min(img_metas[batch_id]['img_shape'][:-1]))]
                    single_kwargs = dict()
                    for key, value in kwargs.items():
                        single_kwargs[key] = [value[batch_id]]
                    # Generate scaled image based on scaled resolution
                    lr_img, lr_img_metas, lr_kwargs = \
                        self.generate_lr_pair(single_img, single_img_metas, single_res, **single_kwargs)
                    with torch.no_grad():
                        rs_output = self.stu.forward_train(lr_img, lr_img_metas, is_train=False, **lr_kwargs)
                        batch_rcg_pred.append(rs_output['rcg_pred'])
                        penalty[batch_id][res_id] = self.policy_cfg.scale_factor[res_id] ** 1.5

                rcg_decisions = [decisions[batch_id] for _ in range(len(gt_texts[batch_id]))]
                rcg_decisions = torch.stack(rcg_decisions).transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
                rs_rcg_pred.append(torch.sum(torch.stack(batch_rcg_pred) * rcg_decisions, dim=0))

            # Calculate the loss of the resolution selector
            rs_rcg_pred = torch.cat(rs_rcg_pred, dim=0)
            loss_rs = self.stu.recog_sequence_head.loss(rs_rcg_pred, recog_target)
            losses.update({"loss_rs": loss_rs['loss_att']})

            loss_penalty = torch.mean(torch.sum(penalty * decisions, dim=-1)) * self.kd_cfg.penalty_coef
            losses.update({"loss_penalty": loss_penalty})

        # Student training procedurek
        if self.policy is not None:
            _, indices = torch.max(decisions, dim=-1)
            indices = indices.cpu().numpy().tolist()
            train_res = [int(self.policy_cfg.scale_factor[indices[i]] *
                             min(img_metas[i]['img_shape'][:-1])) for i in range(len(indices))]
        else:
            train_res = []
            for i in range(img.size(0)):
                scale_idx = np.random.randint(len(self.kd_cfg.scale_factor))
                train_res.append(int(self.kd_cfg.scale_factor[scale_idx] * min(img_metas[i]['img_shape'][:-1])))

        # Generate scaled image based on selected resolution
        lr_img, lr_img_metas, lr_kwargs = self.generate_lr_pair(img, img_metas, train_res, **kwargs)
        stu_output = self.stu.forward_train(lr_img, lr_img_metas, is_train=True, **lr_kwargs)

        stu_rcg_pred = stu_output['rcg_pred']
        stu_roi_feat = stu_output['rcg_roi_feat']
        stu_context_feat = stu_output['rcg_context_feat']

        # Calculate the loss of the distillation
        # Notice: If the teacher's prediction is accurate, it is better to use beam search result
        decode_result = beam_decode(F.softmax(tea_rcg_pred, dim=-1))
        loss_kd_seq = self.seq_loss(stu_rcg_pred, decode_result)
        # loss_kd_seq = self.kl_loss(tea_rcg_pred, stu_rcg_pred)
        losses.update({"loss_kd_seq": loss_kd_seq})

        loss_kd_roi_feat = self.l2_loss(stu_roi_feat, tea_roi_feat)
        losses.update({"loss_kd_roi": loss_kd_roi_feat})

        loss_kd_context_feat = self.l2_loss(stu_context_feat, tea_context_feat)
        losses.update({"loss_kd_context": loss_kd_context_feat})

        # Calculate the loss of the student network
        for key, value in stu_output.items():
            if key not in ['rcg_pred', 'rcg_roi_feat', 'rcg_context_feat', 'recog_target']:
                losses.update({key: value})

        return losses

    def generate_lr_pair(self, img, img_metas, resolution, is_train=True, **kwargs):
        """ Generate low resolution pair image and its information.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            resolution (int): scaled img resolution
            is_train (bool): Whether train mode
            **kwargs: other parameters

        Returns:
            Tensor: low resolution pair image
        Returns:
            dict: image meta infos
        Returns:
            dict: image annotation
        """
        batch_size = img.size(0)
        lr_img = []
        pad_lr_img = []
        lr_img_metas = []
        lr_kwargs = dict()

        max_h = 0
        max_w = 0

        for key in kwargs.keys():
            lr_kwargs[key] = list()

        # Iterate over images
        for batch_id in range(batch_size):
            ori_h, ori_w, _ = img_metas[batch_id]['img_shape']
            batch_img = img[batch_id][:, :ori_h, :ori_w]

            # Calculate scale_factor
            if is_train:
                scale_factor_long = 1600 / max(ori_h, ori_w)
            else:
                scale_factor_long = 3000 / max(ori_h, ori_w)
            scale_factor_short = resolution[batch_id] / min(ori_h, ori_w)
            scale_factor = min(scale_factor_short, scale_factor_long)

            # Resize img
            batch_lr_img = F.interpolate(batch_img.unsqueeze(0),
                                         scale_factor=scale_factor, mode='bilinear').squeeze(0)
            _, new_h, new_w = batch_lr_img.size()
            h_scale = new_h / ori_h
            w_scale = new_w / ori_w
            pad_h = math.ceil(new_h / 32) * 32
            pad_w = math.ceil(new_w / 32) * 32
            max_h = max(max_h, pad_h)
            max_w = max(max_w, pad_w)

            # Modify img_metas
            batch_lr_img_metas = img_metas[batch_id].copy()
            batch_lr_img_metas['scale_factor'] *= np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
            batch_lr_img_metas['img_shape'] = (new_h, new_w, 3)
            batch_lr_img_metas['pad_shape'] = (pad_h, pad_w, 3)

            # Resize boxes
            for key in ['gt_bboxes', 'gt_poly_bboxes']:
                if key in kwargs:
                    lr_bboxes = []
                    gt_bboxes = kwargs[key][batch_id]
                    for box in gt_bboxes:
                        if isinstance(box, torch.Tensor):
                            tmp_box = box.clone()
                        else:
                            tmp_box = box.copy()
                        tmp_box[0::2] = box[0::2] * w_scale
                        tmp_box[1::2] = box[1::2] * h_scale
                        lr_bboxes.append(tmp_box)
                    if len(gt_bboxes) == 0:
                        lr_bboxes = gt_bboxes
                    if key == 'gt_bboxes':
                        lr_bboxes = torch.stack(lr_bboxes)
                    lr_kwargs[key].append(lr_bboxes)

            # Resize masks
            for key in ['gt_masks']:
                if key in kwargs:
                    pad_masks = kwargs[key][batch_id].masks
                    ori_masks = pad_masks[:, :ori_h, :ori_w]
                    if len(ori_masks) == 0:
                        lr_masks = np.empty((0, (new_w, new_h)), dtype=np.uint8)
                    else:
                        lr_masks = np.stack([mmcv.imresize(mask, (new_w, new_h),
                                                           interpolation='nearest') for mask in ori_masks])
                    lr_masks = BitmapMasks(lr_masks, new_h, new_w)
                    pad_lr_masks = lr_masks.pad((pad_h, pad_w), pad_val=0)
                    lr_kwargs[key].append(pad_lr_masks)

            for key in ['gt_texts', 'gt_labels']:
                if key in kwargs:
                    val = kwargs[key][batch_id]
                    lr_kwargs[key].append(val)

            lr_img.append(batch_lr_img)
            lr_img_metas.append(batch_lr_img_metas)

        # Pad image
        for batch_id in range(batch_size):
            batch_lr_img = lr_img[batch_id]
            _, new_h, new_w = batch_lr_img.size()
            batch_pad_img = batch_lr_img.new_zeros((3, max_h, max_w))
            batch_pad_img[:, :new_h, :new_w] = batch_lr_img
            pad_lr_img.append(batch_pad_img)

        pad_lr_img = torch.stack(pad_lr_img)
        return pad_lr_img, lr_img_metas, lr_kwargs

    def kl_loss(self, tea_pred, stu_pred, temperature=5):
        """ kl loss

        Args:
            tea_pred (Tensor): teacher pred result
            stu_pred (Tensor): student pred result
            temperature (int): distillation temperature

        Returns:
            Tensor: kl loss
        """
        kl_loss = 0
        _, time_step, _ = tea_pred.size()
        for ts in range(time_step):
            soft_tea_pred = F.softmax(tea_pred[:, ts, :] / temperature, dim=-1)
            soft_stu_pred = F.log_softmax(stu_pred[:, ts, :] / temperature, dim=-1)
            kl_loss += F.kl_div(soft_stu_pred, soft_tea_pred, reduction='batchmean')
        kl_loss = kl_loss / time_step
        return kl_loss

    def l1_loss(self, tea_pred, stu_pred, temperature=5):
        """ l1 loss

        Args:
            tea_pred (Tensor): teacher pred result
            stu_pred (Tensor): student pred result
            temperature (int): distillation temperature

        Returns:
            Tensor: l1 loss
        """
        tea_pred = tea_pred / temperature
        stu_pred = stu_pred / temperature
        l1_loss = F.l1_loss(tea_pred, stu_pred)
        l1_loss = l1_loss * (temperature ** 2)
        return l1_loss

    def l2_loss(self, tea_pred, stu_pred, temperature=5):
        """ l2 loss

        Args:
            tea_pred (Tensor): teacher pred result
            stu_pred (Tensor): student pred result
            temperature (int): distillation temperature

        Returns:
            Tensor: l2 loss
        """
        tea_pred = tea_pred / temperature
        stu_pred = stu_pred / temperature
        l2_loss = F.mse_loss(tea_pred, stu_pred)
        l2_loss = l2_loss * (temperature ** 2)
        return l2_loss

    def seq_loss(self, stu_pred, beam_decode):
        """ seq loss

        Args:
            stu_pred (Tensor): student pred result
            beam_decode (list(Tensor)): beam search result

        Returns:
            Tensor: seq loss
        """
        seq_loss = 0
        soft_stu_pred = F.softmax(stu_pred, dim=-1)
        for batch_id in range(soft_stu_pred.size(0)):
            for path_id in range(len(beam_decode[batch_id])):
                logit = 1
                for char_pos, char_id in enumerate(beam_decode[batch_id][path_id][1:]):
                    logit *= soft_stu_pred[batch_id][char_pos][char_id]
                seq_loss += -torch.log(logit + 1e-5)
        seq_loss = seq_loss / soft_stu_pred.size(0) / len(beam_decode[0])
        return seq_loss

    def simple_test(self,
                    img,
                    img_metas,
                    **kwargs):
        """ Forward test process.

        Args:
            img (Tensor): input images
            img_metas (dict): image meta infos
            **kwargs: other parameters

        Returns:
            dict: formated inference results
        """
        decisions = self.policy(img)
        decisions = F.gumbel_softmax(decisions, tau=1, hard=True)
        _, indices = torch.max(decisions, dim=-1)
        indices = indices.cpu().numpy().tolist()
        train_res = [int(self.policy_cfg.scale_factor[indices[i]] *
                         min(img_metas[i]['img_shape'][:-1])) for i in range(len(indices))]
        lr_img, lr_img_metas, lr_kwargs = self.generate_lr_pair(img, img_metas, train_res, is_train=False, **kwargs)
        results = self.stu.simple_test(lr_img, lr_img_metas, **lr_kwargs)
        return results

    def forward_dummy(self,
                      img,
                      **kwargs):
        """ Forward dummy for calculate flops.

        Args:
            img (Tensor): input images
            **kwargs: other parameters

        Returns:
            dict: dummy output
        """
        decisions = self.policy(img)
        decisions = F.gumbel_softmax(decisions, tau=0.01, hard=True)
        _, indices = torch.max(decisions, dim=-1)
        scale_factor = self.policy_cfg.scale_factor[indices[0]]
        img = F.interpolate(img, scale_factor=scale_factor, mode='bilinear').squeeze(0)
        _, new_h, new_w = img.size()
        pad_h = math.ceil(new_h / 32.0) * 32
        pad_w = math.ceil(new_w / 32.0) * 32
        pad_img = img.new_zeros((1, 3, pad_h, pad_w))
        pad_img[:, :, :new_h, :new_w] = img
        outs = self.stu.forward_dummy(pad_img)
        return outs
