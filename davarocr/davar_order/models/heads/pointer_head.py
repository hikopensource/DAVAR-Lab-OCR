"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    pointer_head.py
# Abstract       :    pointer-net head used in gcn-pn.

# Current Version:    1.0.0
# Date           :    2022-05-11
######################################################################################################
"""
import numpy as np
import math

from mmdet.models.builder import HEADS, build_loss
import torch
import torch.nn as nn


@HEADS.register_module()
class PointerHead(nn.Module):
    """PointerHead implementation."""

    def __init__(self,
                 init_channel=256,
                 query_in_channel=768,
                 query_out_channel=256,
                 key_in_channel=256,
                 key_out_channel=256,
                 loss=dict(type='StandardCrossEntropyLoss')):
        """
        Args:
            init_channel (int): init channel number for decoder.
            query_in_channel (int): input query channel number.
            query_out_channel (int): output query channel number.
            key_in_channel (int): input key channel number.
            key_out_channel (int): output key channel number.
            loss (dict): loss config.
        """
        super().__init__()
        self.decoder_init = nn.Parameter(torch.FloatTensor(init_channel))
        self.hidden = nn.Parameter(torch.FloatTensor(init_channel))
        self.query = nn.Linear(query_in_channel, query_out_channel)
        self.key = nn.Linear(key_in_channel, key_out_channel)
        self.softmax = nn.Softmax(dim=1)
        self.loss_cls = build_loss(loss)
        self.dec = nn.LSTMCell(256, 256)

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        return

    def forward(self, input_feat):
        """ Forward computation

        Args:
            input_feat (Tensor): in shape of [B x L x M], where M is the dimension of features.
        Returns:
            Tensor: in shape of [B x L x D], where D is the num_classes.
        """
        pred = self.fc_logits(input_feat)
        return pred

    def get_target(self, gt_labels):
        """

        Returns:
            Tensor: gt labels, just return the inputs without any manipulation.
        """
        return gt_labels


    def get_predict(self, batch_z_g, batch_z_l, batch_node_embedding):
        """ get the final predictions.

        Args:
            batch_z_g (list(Tensor)):
                global feature vector, in shape of B x M, where M is the dimension of features
            batch_z_l (list(Tensor)):
                local feature vector, in shape of B x M, where M is the dimension of features
            batch_node_embedding (list(Tensor)):
                in shape of B x N x M, where M is the dimension of features, N is all text bboxes in a batch

        Returns:
            list: in shape of [N], decoding labels of pred.
        """
        batch_size = len(batch_z_g)
        batch_orders = []
        for batch in range(batch_size):
            orders = []
            # inputs
            node_embedding = batch_node_embedding[batch]
            z_g = batch_z_g[batch]
            z_l = batch_z_l[batch]
            index = 0
            seq_length = node_embedding.shape[0]

            hidden = self.hidden.unsqueeze(0)
            cell_state = z_g.unsqueeze(0)
            key = self.key(node_embedding)
            while index < seq_length:
                hidden, cell_state = self.dec(self.decoder_init.unsqueeze(0), (hidden, cell_state))
                query = self.query(hidden)
                
                #1*len(labels)
                attention_scores = torch.matmul(query, key.transpose(-1, -2))
                mask = torch.zeros_like(attention_scores)

                # mask output in previous time step
                if orders:
                    mask[:,orders] = -1e9
                    attention_scores += mask
                
                # get attention scores
                attention_scores = self.softmax(attention_scores)

                cur_index = np.argmax(attention_scores.view(-1).detach().cpu().numpy())
                orders.append(cur_index)
                index += 1

            batch_orders.append(orders)
        return batch_orders

    def loss(self, batch_z_g, batch_z_l, batch_node_embedding, gt_labels):
        """ loss computation.

        Args:
            batch_z_g (list(Tensor)):
                global feature vector, in shape of B x M, where M is the dimension of features
            batch_z_l (list(Tensor)):
                local feature vector, in shape of B x M, where M is the dimension of features
            batch_node_embedding (list(Tensor)):
                in shape of B x N x M, where M is the dimension of features, N is all text bboxes in a batch
            gt_labels (list(Tensor)): in shape of B x N, where N is all text bboxes in a batch

        Returns:
            tensor: loss value.
        """
        total_loss = 0
        batch_size = len(batch_z_g)
        for batch in range(batch_size):
            # input features
            node_embedding = batch_node_embedding[batch]
            z_g = batch_z_g[batch]
            z_l = batch_z_l[batch]
            gt_label = gt_labels[batch]
            seq_length = node_embedding.shape[0]
            hidden = self.hidden.unsqueeze(0)
            cell_state = z_g.unsqueeze(0)
            key = self.key(node_embedding)

            # compute attention scores
            attention_scores = []
            for i in range(seq_length):
                # lstm to get hidden state
                hidden, cell_state = self.dec(self.decoder_init.unsqueeze(0), (hidden, cell_state))
                query = self.query(hidden.view(1,-1))
                attention_score = torch.matmul(query, key.transpose(-1, -2))
                attention_scores.append(attention_score)
            attention_scores = torch.cat(attention_scores,dim=0)
            d_h = key.shape[1]
            attention_scores /= math.sqrt(d_h)

            device = node_embedding.device

            # mask output in previous time step
            attention_mask = torch.tril(torch.ones(seq_length,seq_length),-1)
            attention_mask = attention_mask.to(device)

            index = [(gt_label-1).cpu().numpy().tolist().index(i) for i in range(len(gt_label))]
            attention_mask = attention_mask[:,index]

            attention_mask = attention_mask.to(dtype=torch.float32)  # fp16 compatibility
            attention_mask *= -1e9

            #seq_len*len(labels)
            attention_scores += attention_mask

            # get final attention scores
            attention_scores = self.softmax(attention_scores)

            # compute loss
            loss = 0
            for i in range(seq_length):
                p = attention_scores[i,(gt_label-1)[i]]
                loss += -torch.log(p+1e-7)

            total_loss += loss

        return total_loss/batch_size

            

            
