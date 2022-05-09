"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    multi_recog_seq_head.py
# Abstract       :    Recognize text in a batch.

# Current Version:    1.0.0
# Date           :    2021-03-19
######################################################################################################
"""
import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner import auto_fp16
from davarocr.davar_common.core.builder import build_converter


@HEADS.register_module()
class MultiRecogSeqHead(nn.Module):
    """ Implementation of multi-batch character classification in MANGO [1].

    Ref: [1] MANGO: A Mask Attention Guided One-Staged Text Spotter. AAAI-21.
             <https://arxiv.org/abs/2012.04350>`_
    """

    def __init__(self,
                 in_channels=256,
                 fc_out_channels=512,
                 text_max_length=25,
                 featmap_indices=(0, 1, 2, 3),
                 num_fcs=2,
                 converter=None,
                 loss_recog=None,
                 ):
        """
        Args:
            in_channels (int): input feature map channels.
            fc_out_channels (int): output feature map channels.
            text_max_length (int): the max length of recognition words.
            featmap_indices (list(int)): selected feature map scales.
            num_fcs (int): stacked FC layers number
            converter (dict): converter types
            loss_recog (dict): recognition loss
        """

        super().__init__()
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.text_max_length = text_max_length
        self.num_fcs = num_fcs
        self.featmap_indices = featmap_indices
        assert converter is not None
        self.converter = build_converter(converter)
        self.num_classes = len(self.converter.character)

        if loss_recog is not None:
            self.loss_recog = build_loss(loss_recog)
        else:
            self.loss_recog = None

        self.fcs = self._add_fc_branch(self.in_channels, self.num_fcs)
        self.fc_logits = nn.Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        """ Weight initialization. """
        for fc_layer in self.fcs:
            if isinstance(fc_layer, nn.Linear):
                nn.init.xavier_uniform_(fc_layer.weight)
                nn.init.constant_(fc_layer.bias, 0)

    def _add_fc_branch(self,
                       in_channels,
                       num_branch_fcs
                       ):
        """ Generate stack fully-connected layers

        Args:
            in_channels (int): input channel number
            num_branch_fcs (int): how many fc layers

        Returns:
            nn.ModuleList: stacked fc layers
        """

        last_layer_dim = in_channels
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            for i in range(num_branch_fcs):
                fc_in_channels = (last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def forward_single(self, feats, matched_bboxes):
        """ Predicting characters based on features map in single level.

        Args:
            feats (Tensor): input feature map in shape of [BKL, C]
            matched_bboxes (Tensor): grid category, in shape of [B, S^2]

        Returns:
            Tensor: prediction results, in shape of [B, K, L, num_class]
        """

        max_category_num = torch.max(torch.sum(torch.ge(matched_bboxes,1),
                                               dim=1))
        if int(max_category_num) < 1:
            return None
        for fc_layer in self.fcs:
            feats = self.relu(fc_layer(feats))
        pred = self.fc_logits(feats).view(matched_bboxes.shape[0], max_category_num, self.text_max_length,
                                          self.num_classes)
        return pred


    @auto_fp16()
    def forward(self, feats, matched_bboxes):
        """ Predicting characters based on features map in single level.

        Args:
            feats (list(Tensor)): input feature map in shape of [BKL, C]
            matched_bboxes (list(Tensor)): grid category, in shape of [B, S^2]

        Returns:
            Tensor: prediction results, in shape of [B, K, L, num_class]
        """

        preds = []
        for i in range(len(self.featmap_indices)):
            new_feat = feats[i].contiguous().view(-1, feats[i].shape[-1])
            pred = self.forward_single(new_feat, matched_bboxes[i])
            preds.append(pred)
        return preds

    def get_target_single(self,
                          gt_texts,
                          matched_bboxes,
                          feat_size,
                          device='cuda'):
        """ Generate training target according to word level annotations in single level.

        Args:
            gt_texts (list(string)): A variable length list [K, *]
            matched_bboxes (Tensor): grid category, a tensor of shape [B, S^2]
            feat_size (tuple): feature map shape
            device (str): running device

        Returns:
            Tensor: the training target, in shape of [B, K, L]
        """

        batch = feat_size[0]
        max_category_num = torch.max(torch.sum(torch.ge(matched_bboxes,1), dim=1))
        gt_label = torch.zeros([batch, max_category_num, self.text_max_length], dtype=torch.long, device=device)
        values, _ = torch.topk(matched_bboxes, max_category_num)  # B x K
        for batch_id in range(batch):
            val = values[batch_id]
            gt_text = gt_texts[batch_id]
            # Convert string into int
            encode_text, _ = self.converter.encode(gt_text, self.text_max_length)

            for idx, text in enumerate(encode_text):
                indices = torch.where(val == idx + 1)[0]
                for ind in indices:
                    gt_label[batch_id, ind, ...] = text.unsqueeze(0)
        return gt_label

    def get_target(self, feats, gt_texts, matched_bboxes):
        """ Generate training target according to word level annotations in multiple levels

        Args:
            feat (list(Tensor)): input feature maps, in shape of [B, K, C]
            gt_texts (list(string)): A variable length list [K, *]
            matched_bboxes (list(Tensor)): grid category, a tensor of shape [B, S^2]

        Returns:
           list(Tensor): the training target, in shape of [B, K, L]
        """

        targets = []
        for i, _ in enumerate(self.featmap_indices):
            target = self.get_target_single(
                gt_texts,
                matched_bboxes[i],
                feats[i].shape,
                device=feats[i].device
            )
            targets.append(target)
        return targets

    def accuracy_metric(self, pred, target, stride, stop_token=1):
        """ Calculate the batch accuracy in both character level and word level.

        Args:
            pred (list(Tensor)): prediction text sequences, in shape of [B, K, L]
            target (list(Tensor)): target text sequences , in shape of [B, K, L]
            stride (int): feature map stride, e.g. 4
            stop_token (int): EOS label index

        Returns:
            dict: accuracies in a dict.
        """

        pred_id = torch.argmax(pred, dim=-1)
        stop_inds = torch.nonzero(target == stop_token)
        correct = pred_id.eq(target)
        char_correct = sum([torch.sum(correct[ind[0], ind[1], :ind[2]]) for ind in stop_inds]).float()
        char_total = sum(stop_inds[:, 2])
        string_correct = sum([torch.sum(correct[ind[0], ind[1], :ind[2]]) == ind[2] for ind in stop_inds]).float()
        string_total = len(stop_inds)
        acc_metric = dict()
        acc_metric.update({"acc_char_{}x".format(stride): char_correct / char_total})
        acc_metric.update({"acc_str_{}x".format(stride): string_correct / string_total})
        return acc_metric

    def loss(self, text_preds, text_targets):
        """ Loss computation.

        Args:
            text_preds (list(Tensor)): text predictions, in shape of [B, K, L]
            text_targets (list(Tensor)): text targets, in shape of [B, K, L]

        Returns:
            dict: losses in a dict.
        """

        loss = dict()
        for i, stride_idx in enumerate(self.featmap_indices):
            stride = 4 * (2 ** stride_idx)
            text_pred = text_preds[i]
            if text_pred is None:
                continue
            gt_text = text_targets[i]
            loss_recog = self.loss_recog(text_pred.view(-1, text_pred.size(-1)), gt_text.view(-1))
            loss.update({"loss_recog_{}x".format(stride):loss_recog})
            loss.update(self.accuracy_metric(text_pred, gt_text, stride))
        return loss

    def get_pred_text(self, preds):
        """ Decode predicted label into original characters

        Args:
            preds (list(Tensor)): predict labels, in shape of [B, K, L]

        Returns:
            list(str): predict strings.
        """

        batch_preds = []
        for pred in preds:
            pred = pred.squeeze(0)
            batch_size = pred.size(0)
            length_for_pred = torch.cuda.IntTensor([self.text_max_length] *
                                                   batch_size)
            pred = pred[:, :self.text_max_length, :]
            _, preds_index = pred.max(2)
            preds_index = preds_index.contiguous()
            preds_str = self.converter.decode(preds_index, length_for_pred)
            for i, tmp_str in enumerate(preds_str):
                if "[s]" in tmp_str:
                    preds_str[i] = tmp_str[:tmp_str.find('[s]')]
            batch_preds.append(preds_str)
        return batch_preds
