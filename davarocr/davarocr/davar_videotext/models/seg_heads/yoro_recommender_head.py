"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    yoro_recommender_head.py
# Abstract       :    Implementations of the Attn-based recognition branch, track branch, qality score branch

# Current Version:    1.0.0
# Date           :    2021-06-06
##################################################################################################
"""

import logging

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmdet.models.builder import HEADS
from mmdet.models.builder import build_loss
from davarocr.davar_rcg.models.sequence_heads import AttentionHead
from davarocr.davar_rcg.models.sequence_heads.att_head import AttentionCell


@HEADS.register_module()
class TextRecommenderHead(AttentionHead):
    """Text Recommender head structure, This head is used for recognition, tracking, scoring according to ground-truth
    labels.

    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_max_length=25,
                 converter=dict(
                     type='AttnLabelConverter',
                     character='0123456789abcdefghijklmnopqrstuvwxyz', ),
                 loss_att=dict(
                    type='CELoss',
                    ignore_index=-100,
                    weight=1.0,
                    reduction='mean'),
                 loss_triplet=dict(
                     type='TripletLoss',
                     margin=0.5,
                     reduction='mean'),
                 loss_l1=dict(
                     type='L1Loss',
                     reduction='mean'
                 ),
                 head_cfg=None
                 ):
        """ Text Recommender head structure.
        Args:
            input_size (int): input size
            hidden_size (int): hidden  state size
            batch_max_length (int): batch max text length
            converter (dict): converter parameter
            loss_att (dict): loss function for recognition
            loss_triplet(dict): loss function for tracking
            loss_score(dict): loss function for scoring
        """
        super().__init__(input_size, hidden_size, batch_max_length, loss_att, converter)

        self.head_cfg = head_cfg

        # Attention cell
        attention_cell_num = self.num_classes
        self.attention_cell = CustomAttentionCell(input_size, hidden_size, attention_cell_num)


        self.generator = nn.Linear(hidden_size, self.num_classes)

        # If train track branch or quality branch, It is better when fix the recognition branch
        if self.head_cfg and self.head_cfg.get('fix_rcg', False):
            for p in self.parameters():
                p.requires_grad = False

        # Track branch
        self.track_branch = TrackBranch(hidden_size)

        # Build the track branch loss
        self.loss_triplet = build_loss(loss_triplet)

        # If train quality score branch, the quality branch is different from recognition and track branch, It needs
        # not care samples to learn the low quality distribution. So when train quality branch, we'd better fix the
        # track branch and recognition branch
        if self.head_cfg and self.head_cfg.get('fix_track', False):
            self.track_branch.eval()

        # Quality score branch
        self.qscore_branch = QscoreBranch(hidden_size)

        # Build the score branch loss
        self.loss_l1 = build_loss(loss_l1)

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): model path of the pre_trained model

        Returns:

        """
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            logger.info("AttentionHead:")
            load_checkpoint(self, pretrained, strict=False, logger=logger)

        elif pretrained is None:
            for name, param in self.named_parameters():
                if 'track_branch' in name or 'qscore_branch' in name:
                    if 'bn' in name and 'bias' in name:
                        init.constant_(param, 0.0)
                    elif 'bn' in name and 'weight' in name:
                        init.constant_(param, 1)
                    elif 'bias' in name:
                        init.constant_(param, 0.0)
                    elif 'weight' in name:
                        init.kaiming_normal_(param)
                elif 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)

    def forward(self, batch_H, visual_feature, target, is_train=True):
        """
        Args:
            batch_H : contextual_feature H = hidden state of encoder.
            [batch_size x num_steps x num_classes]
            visual_feature(Tensor): lower feature foi tracking and quality score
            [batch_size x hidden_size x H/4 x W/4]
            target (tensor): label information
            is_train (bool): whether is training state or test state

        Returns:
            Tensor: probability distribution at each step

        Returns:
            Tensor: discriminative track features

        Returns:
            Tensor: quality scores
        """
        if self.head_cfg and self.head_cfg.get('fix_track', False):
            self.track_branch.eval()

        # Track branch
        track_feature = self.track_branch(visual_feature)

        # Qscore branch
        pred_scores = self.qscore_branch(visual_feature)

        # Recognition branch
        if target is not None:
            gt_label = target
        else:
            gt_label = None
        batch_size, _, input_size = batch_H.size()

        # +1 for [s] at end of sentence.
        num_steps = self.batch_max_length + 1

        output_hiddens = torch.cuda.FloatTensor(batch_size,
                                                num_steps,
                                                self.hidden_size).fill_(0)
        hidden = (torch.cuda.FloatTensor(batch_size, self.hidden_size,
                                         device=batch_H.device).fill_(0),
                  torch.cuda.FloatTensor(batch_size, self.hidden_size,
                                         device=batch_H.device).fill_(0))

        glimpses = torch.cuda.FloatTensor(batch_size, num_steps, input_size)

        if is_train:
            for i in range(num_steps):

                # The vector corresponding to the i-th text in one batch
                char_onehots = self._char_to_onehot(gt_label[:, i], onehot_dim=self.num_classes)

                # Hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, _, glimpse = self.attention_cell(hidden, batch_H, char_onehots)

                # LSTM hidden index (0: hidden, 1: Cell)
                output_hiddens[:, i, :] = hidden[0]
                glimpses[:, i, :] = glimpse

            probs = self.generator(output_hiddens)
        else:

            # [GO] token
            targets = torch.cuda.LongTensor(batch_size,
                                            device=batch_H.device).fill_(self.bos)
            probs = torch.cuda.FloatTensor(batch_size,
                                           num_steps,
                                           self.num_classes,
                                           device=batch_H.device).fill_(0)
            for i in range(num_steps):

                # The vector corresponding to the i-th text in one batch
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)

                # Hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, _, glimpse = self.attention_cell(hidden, batch_H, char_onehots)

                # LSTM hidden index (0: hidden, 1: Cell)
                output_hiddens[:, i, :] = hidden[0]

                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)

                targets = next_input
                glimpses[:, i, :] = glimpse

        # The glimpse features is used for generating gt quality scores
        if not is_train:
            return probs, glimpses, track_feature, pred_scores

        return probs, track_feature, pred_scores



    def loss(self, preds, target):
        """

        Args:
            preds (tuple): predict texts, discriminative track feature, predict quality scores
            target (tuple): gt texts, length, gt scores

        Returns:
            model training loss

        """

        pred_texts, track_feature, pred_scores = preds

        # Make sure that the features is made of three parts: anchor, positive, negative
        batch_size = track_feature.size(0)
        assert batch_size % 3 == 0
        anchor_nums = batch_size // 3

        # Fetch anchor, positive, negative features for track
        anchor = track_feature[:anchor_nums]
        positive = track_feature[anchor_nums:anchor_nums*2]
        negative = track_feature[anchor_nums*2:]

        pred_scores = pred_scores.squeeze()

        loss = dict()

        # Fetch gt texts, gt quality scores
        gt_label, _, gt_scores = target

        # without [GO] Symbol shape
        gt_label = gt_label[:, 1:]

        # Only fix_rcg is False, can we train the rcg branch
        if not (self.head_cfg and self.head_cfg.get('fix_rcg', False)):
            loss_att = self.loss_att(pred_texts.view(-1, pred_texts.shape[-1]),
                                     gt_label.contiguous().view(-1))
            loss['loss_att'] = loss_att

        # Only fix_track is False, can we train the track branch
        if not (self.head_cfg and self.head_cfg.get('fix_track', False)):

            loss_triplet = self.loss_triplet(anchor, positive, negative)
            print('anchor-positive sim', torch.cosine_similarity(anchor, positive, dim=1))
            print('anchor-negative sim', torch.cosine_similarity(anchor, negative, dim=1))
            loss['loss_triplet'] = loss_triplet

        # Only fix_qscore is False, can we train the score branch
        if not (self.head_cfg and self.head_cfg.get('fix_qscore', False)):
            loss['loss_l1'] = self.loss_l1(pred_scores, gt_scores)
            print('gt  scores', gt_scores[:10])
            print('pre scores', pred_scores[:10])

        return loss

    def get_target(self, gts):
        """
        Args:
            gts (tuple): gt texts, img metas

        Returns:
            vector transformed by text label and converter

        Returns:
            length vector transformed by text label and converter

        Returns:
            Tensor: gt quality scores
        """
        gt_texts, img_metas = gts

        # Fetch gt quality scores
        scores = torch.cuda.FloatTensor(len(img_metas))
        for i, data in enumerate(img_metas):
            img_info = data['img_info']
            score = img_info['ann']['score']
            if not score:
                break
            scores[i] = score

        # Fetch gt texts
        if gt_texts is not None:
            text, length = self.converter.encode(gt_texts, self.batch_max_length)
            return text, length, scores

        return None, None, None


class CustomAttentionCell(AttentionCell):
    """ Attention Cell Structure """
    def __init__(self, input_size, hidden_size, num_embeddings):

        """
        Args:
            input_size (int): input channel
            hidden_size (int): hidden state num
            num_embeddings (int): embedding layers
        """
        super().__init__(input_size=input_size, hidden_size=hidden_size, num_embeddings=num_embeddings)

    def forward(self, prev_hidden, batch_h, char_onehots):
        """
        Args:
            prev_hidden (tensor): previous layer's hidden state
            batch_h (tensor): sequential input feature
            char_onehots (tensor): one hot vector

        Returns:
            current hidden state

        Returns:
            attention weight

        Returns:
            glimpse features

        """
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_h_proj = self.i2h(batch_h)

        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)

        # [batch_size x num_encoder_step * 1]
        emphasis = self.score(torch.tanh(batch_h_proj + prev_hidden_proj))

        alpha = F.softmax(emphasis, dim=1)

        # [batch_size x num_channel]
        context = torch.bmm(alpha.permute(0, 2, 1), batch_h).squeeze(1)

        glimpse = context

        # [batch_size x (num_channel + num_embedding)]
        concat_context = torch.cat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha, glimpse


class QscoreBranch(nn.Module):
    """ Quality Score branch in Text Recommender head Structure """
    def __init__(self,
                 hidden_size
                 ):
        """
        Args:
            hidden_size (int): feature middle hidden channels nums
        """
        super().__init__()
        self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(self.input_size,
                               self.hidden_size,
                               kernel_size=(3,3), stride=2,
                               padding=0)
        self.bn1 = nn.BatchNorm2d(self.hidden_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_size * 3 * 12, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): lower contextual_feature [batch x hidden size x H/4 x W/4]

        Returns:
            Torch.Tensor: predict quality scores

        """
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(batch_size, -1).contiguous()
        x = self.fc1(x)
        scores = self.sigmoid(x)
        return scores


class TrackBranch(nn.Module):
    """ Track branch in Text Recommender head Structure """
    def __init__(self,
                 hidden_size
                 ):
        """
        Args:
            hidden_size (int): hidden state num
        """
        super().__init__()
        self.input_size = hidden_size
        self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(self.input_size,
                               self.hidden_size,
                               kernel_size=(3,3), stride=1,
                               padding=0)
        self.bn1 = nn.BatchNorm2d(self.hidden_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(6 * 23, 1)

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): lower contextual_feature [batch x hidden size x H/4 x W/4]

        Returns:
            Torch.Tensor: the learned discriminative track feature

        """

        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(batch_size * self.hidden_size, -1)
        x = self.fc1(x)
        x = x.view(batch_size, -1)
        return x
