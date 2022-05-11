"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    tps_transformation.py
# Abstract       :    Implementations of the TPS transformation

# Current Version:    1.0.0
# Date           :    2021-03-07
# Thanks to      :    We borrow the released code from http://gitbug.com/clovaai/deep-text-recognition-benchmark
                      for the TPS transformer.
##################################################################################################
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from davarocr.davar_common.models.builder import TRANSFORMATIONS


@TRANSFORMATIONS.register_module()
class TPS_SpatialTransformer(nn.Module):
    """ Rectification Network of RARE, namely TPS-based STN

    Ref: [1] Spatial Transformer Network, NIPS-2016
         [2] Robust scene text recognition with automatic rectification. CVPR-2016. (short in RARE)
         [3] ASTER: An Attentional Scene Text Recognizer with Flexible Rectification. T-PAMI 2018.

    Usage Example:
    transformation=dict(
         type='TPS_SpatialTransformer',
         F=20,
         I_size=(32, 100),
         I_r_size=(32, 100),
         I_channel_num=3),
    """

    def __init__(self,
                 F,
                 I_size,
                 I_r_size,
                 I_channel_num=1):
        """

        Args:
            F (int): number of fiducial points (default 20 following the paper)
            I_size (tuple): size of input images
            I_r_size (tuple): size of rectified images
            I_channel_num (int): the number of channels of the input image I
        """

        super(TPS_SpatialTransformer, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.I_channel_num = I_channel_num

        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F,
                                           self.I_r_size)

    def forward(self, batch_I):
        """
        Args:
            batch_I (tensor): batch of input images [batch_size x I_channel_num x I_r_height x I_r_width]

        Returns:
            np.array: the image of rectified images

        """

        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)  # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0),
                                                       self.I_r_size[0],
                                                       self.I_r_size[1],
                                                       2])
        batch_I_r = F.grid_sample(batch_I,
                                  build_P_prime_reshape,
                                  padding_mode='border')

        return batch_I_r

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): save path of pretrained model

        Returns:

        """
        return


class LocalizationNetwork(nn.Module):
    """ Localization Network of RARE,
    which predicts C' (K x 2) from I (I_width x I_height) """

    def __init__(self, F, I_channel_num):
        """

        Args:
            F (int): number of fiducial points (default 20 following the paper)
            I_channel_num (int): the number of channels of the input image I
        """
        super(LocalizationNetwork, self).__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.I_channel_num,
                      out_channels=64, kernel_size=3,
                      stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1,
                      bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            nn.Conv2d(128, 256, 3, 1, 1,
                      bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            nn.Conv2d(256, 512, 3, 1, 1,
                      bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256),
                                              nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)

        # see RARE paper Fig. 6 (a)
        ctrl_pts_x = np.linspace(-1.0, 1.0,
                                 int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0,
                                     num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0,
                                        num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top],
                                axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom],
                                   axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom],
                                      axis=0)
        self.localization_fc2.bias.data = torch.from_numpy(initial_bias).float().view(-1)

    def forward(self, batch_I):

        """
        Args:
            x (tensor): input image featuremaps [batch_size x I_channel_num x I_height x I_width]

        Returns:
            torch.Tensor: Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)). \
            view(batch_size, self.F, 2)
        return batch_C_prime


class GridGenerator(nn.Module):
    """ Grid Generator of RARE,
    which produces P_prime by multipling T with P

    """

    def __init__(self, F, I_r_size):
        """
        Args:
            F (int): number of fiducial points (default 20 following the paper)
            I_r_size (tuple): size of rectified images
        """
        """ Generate P_hat and inv_delta_C for later """
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.register_buffer("inv_delta_C",  # F+3 x F+3
                             torch.tensor(self._build_inv_delta_C(self.F, self.C)).float())

        self.register_buffer("P_hat",
                             torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float())  # n x F+3

    def _build_C(self, F):
        """
        Args:
            F (int): number of fiducial points (default 20 following the paper)

        Returns:
            np.array: coordinates of fiducial points in I_r [batch_size x F x 2]

        """
        """ Return coordinates of fiducial points in I_r; C """
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top],
                                axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom],
                                   axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom],
                           axis=0)
        return C

    def _build_inv_delta_C(self, F, C):
        """

        Args:
            F (int): number of fiducial points (default 20 following the paper)
            C (np.array): coordinates of fiducial points

        Returns:
            np.array: TPS transformation matrix [(F+3) x (F+3)]

        """
        # Return inv_delta_C which is needed to calculate T
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)

        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)),
                                C, hat_C], axis=1),         # F x F+3
                np.concatenate([np.zeros((2, 3)),
                                np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)),
                                np.ones((1, F))], axis=1)   # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        """
        Args:
            I_r_width (torch.Tensor): width of rectified images
            I_r_height (torch.Tensor): height of rectified images

        Returns:
            np.array: generated P [(I_r_width x I_r_height) x 2]

        """

        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width     # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height

        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x,
                        I_r_grid_y), axis=2)
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        """

        Args:
            F (int): number of fiducial points
            C (np.array): coordinates of fiducial points

            P (np.array): the generated sampling grid P on I

        Returns:

        """
        n = P.shape[0]                                  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1),
                         (1, F, 1))                     # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)              # 1 x F x 2
        P_diff = P_tile - C_tile                        # n x F x 2
        rbf_norm = np.linalg.norm(P_diff,
                                  ord=2,
                                  axis=2,
                                  keepdims=False)       # n x F
        rbf = np.multiply(np.square(rbf_norm),
                          np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """

        Args:
            batch_C_prime (tensor):

        Returns:
            torch.Tensor: generated grid [batch_size x F x 2]

        """
        # Generate Grid from batch_C_prime [batch_size x F x 2]
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime,
                                              torch.zeros(batch_size, 3, 2).float().cuda()),
                                             dim=1)    # batch_size x
        # (F+3) x 2
        batch_T = torch.bmm(batch_inv_delta_C,
                            batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat,
                                  batch_T)             # batch_size x n x 2
        return batch_P_prime                           # batch_size x n x 2
