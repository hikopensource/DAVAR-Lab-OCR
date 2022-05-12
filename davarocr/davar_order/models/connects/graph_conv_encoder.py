"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    graph_conv_encoder.py
# Abstract       :    encode feature for each bbox/ node.

# Current Version:    1.0.0
# Date           :    2022-05-11
######################################################################################################
"""
import torch
import torch.nn as nn

from davarocr.davar_common.models import CONNECTS


class EdgeAttention(nn.Module):
    """Implementation of edge attention in GCN-PN

    Ref: An End-to-End OCR Text Re-organization Sequence Learning for Rich-text Detail Image Comprehension. ECCV-20.
    """
    def __init__(self,in_channel=256):
        """
        Args:
            in_channel (int): input channel number
        """
        super(EdgeAttention, self).__init__()
        self.conv = nn.Conv2d(in_channel, 1, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, edge_feats):
        """ Forward computation

        Args:
            edge_feats (Tensor):
                features for attention map, in shape of 1 x M x N x N, where N is all text bboxes in a batch,
                M is the dimension of features

        Returns:
            Tensor: updated node features, in shape of [1 x N x M], where N is all text bboxes in a batch,
                    M is the dimension of features
        """
        attention_w = self.leaky_relu(self.conv(edge_feats))
        attention_coefficient = self.softmax(attention_w)
        node_feats = torch.sum(attention_coefficient*edge_feats,dim=-1)
        return node_feats



@CONNECTS.register_module()
class GraphConvEncoder(nn.Module):
    """Implementation of encoder in GCN-PN

    Ref: An End-to-End OCR Text Re-organization Sequence Learning for Rich-text Detail Image Comprehension. ECCV-20.
    """
    def __init__(self,
                 graph_conv_block_num=2,
                 conv_layer_num=2,
                 in_channel=256*3,
                 output_channel=256,
                 ):
        """
        Args:
            graph_conv_block_num (int): number of graph convolution layer
            conv_layer_num (int): number of convolution per layer
            in_channel (int): input channel number
            output_channel (int): output channel number
        """
        super(GraphConvEncoder, self).__init__()
        self.graph_conv_block_num = graph_conv_block_num
        self.conv = nn.ModuleList([])
        for i in range(conv_layer_num):
            if i == 0:
                self.conv.append(nn.Conv2d(in_channel, output_channel,kernel_size=3,padding=1))
                self.conv.append(nn.ReLU())
            else:
                self.conv.append(nn.Conv2d(output_channel, output_channel,kernel_size=3,padding=1))
                self.conv.append(nn.ReLU())
        self.last_conv = nn.Sequential(
            nn.Conv2d(output_channel, output_channel,kernel_size=3,padding=1),
            nn.ReLU()
            )
        self.att = EdgeAttention(output_channel)
        self.adapter = nn.Conv2d(7, output_channel,kernel_size=1)

    def init_weights(self, pretrained=None):
        """ Weight initialization

        Args:
            pretrained (str, optional): Path to pre-trained weights. Defaults to None.
        """
        return

    def forward(self,
                batch_edge_feats,
                batch_node_feats):
        """ Forward computation

        Args:
            batch_edge_feats (list(Tensor)):
                position relative edge features, in shape of B x N x N x 7, where N is all text bboxes in a batch
            batch_node_feats (list(Tensor)):
                node features, in shape of B x N x M, where N is all text bboxes in a batch,
                M is the dimension of features

        Returns:
            list(Tensor): updated node features, in shape of B x N x M
        Returns:
            list(Tensor): updated edge features, in shape of B x N x N x M
        """
        batch_size = len(batch_edge_feats)
        graph_conv_block_num = self.graph_conv_block_num
        batch_node_embedding = []
        batch_edge_embedding = []

        # update in a batch
        for batch in range(batch_size):
            edge_feats = batch_edge_feats[batch]
            edge_feats = self.adapter(edge_feats.permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0)
            node_feats = batch_node_feats[batch]
            bbox_num = node_feats.size()[0]
            for i in range(graph_conv_block_num):
                feature = torch.cat([node_feats.unsqueeze(1).repeat(1,bbox_num,1)\
                    ,edge_feats,node_feats.unsqueeze(0).repeat(bbox_num,1,1)],dim=-1)
                feature = feature.permute(2,0,1).unsqueeze(0)
                for module in self.conv:
                    feature = module(feature)
                node_feats = self.att(feature)
                node_feats = node_feats.squeeze(0).permute(1,0)
                edge_feats = self.last_conv(feature)
                edge_feats = edge_feats.squeeze(0).permute(1,2,0)
            batch_node_embedding.append(node_feats)
            batch_edge_embedding.append(edge_feats)
        return batch_node_embedding, batch_edge_embedding

