"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    gaspin_transformation.py
# Abstract       :    Implementations of the gaspin transformation

# Current Version:    1.0.0
# Date           :    2021-03-07
##################################################################################################
"""
import functools
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from davarocr.davar_common.models.builder import TRANSFORMATIONS
from .tps_transformation import GridGenerator
from .spin_transformation import SP_TransformerNetwork


@TRANSFORMATIONS.register_module()
class GA_SPIN_Transformer(nn.Module):
    """
    Geometric-Absorbed SPIN Transformation (GA-SPIN) proposed in Ref. [1]


    Ref: [1] SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition. AAAI-2021.
    """

    def __init__(self, input_channel=3,
                 I_r_size=(32, 100),
                 inputDataType='torch.cuda.FloatTensor',
                 offsets=False,
                 norm_type='BN',
                 default_type=6,
                 rand=False):
        """
        Args:
            input_channel (int): channel of input features,
                                set it to 1 if the grayscale images and 3 if RGB input
            I_r_size (tuple): size of rectified images (used in STN transformations)
            inputDataType (str): the type of input data,
                                only support 'torch.cuda.FloatTensor' this version
            offsets (bool): set it to False if use SPN w.o. AIN,
                            and set it to True if use SPIN (both with SPN and AIN)
            norm_type (str): the normalization type of the module,
                            set it to 'BN' by default, 'IN' optionally
            default_type (int): the K chromatic space,
                                set it to 3/5/6 depend on the complexity of transformation intensities
            rand (bool): initialize type, set it to False by default

        """
        super(GA_SPIN_Transformer, self).__init__()
        self.nc = input_channel
        self.inputDataType = inputDataType
        self.spt = True
        self.offsets = offsets
        self.stn = True  # set to True in GA-SPIN, while set it to False in SPIN
        self.I_r_size = I_r_size
        if norm_type == 'BN':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True,
                                           track_running_stats=True)
        elif norm_type == 'IN':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False,
                                           track_running_stats=False)
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        if self.spt:
            self.sp_net = SP_TransformerNetwork(input_channel,
                                                default_type)
            self.spt_convnet = nn.Sequential(
                                  # 32*100
                                  nn.Conv2d(input_channel, 32, 3, 1, 1,
                                            bias=False),
                                  norm_layer(32), nn.ReLU(True),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  # 16*50
                                  nn.Conv2d(32, 64, 3, 1, 1,
                                            bias=False),
                                  norm_layer(64), nn.ReLU(True),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  # 8*25
                                  nn.Conv2d(64, 128, 3, 1, 1,
                                            bias=False),
                                  norm_layer(128), nn.ReLU(True),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  # 4*12
            )
            self.stucture_fc1 = nn.Sequential(
                                  nn.Conv2d(128, 256, 3, 1, 1,
                                            bias=False),
                                  norm_layer(256), nn.ReLU(True),
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(256, 256, 3, 1, 1,
                                            bias=False),
                                  norm_layer(256), nn.ReLU(True),  # 2*6
                                  nn.MaxPool2d(kernel_size=2, stride=2),
                                  nn.Conv2d(256, 512, 3, 1, 1,
                                            bias=False),
                                  norm_layer(512), nn.ReLU(True),  # 1*3
                                  nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(1, -1),  # batch_size x 512
                                  nn.Linear(512, 256),
                                  nn.BatchNorm1d(256), nn.ReLU(True)
                                )
            self.out_weight = 2*default_type+1
            self.spt_length = 2*default_type+1
            if offsets:
                self.out_weight += 1
            if self.stn:
                self.F = 20
                self.GridGenerator = GridGenerator(self.F,
                                                   (self.I_r_size[0],
                                                    self.I_r_size[1]))
                self.out_weight += self.F * 2

            # self.out_weight*=nc
            self.stucture_fc2 = nn.Linear(256, self.out_weight)
            self.sigmoid = nn.Sigmoid()

            if offsets:
                self.offset_fc1 = nn.Sequential(nn.Conv2d(128, 16,
                                                          3, 1, 1,
                                                          bias=False),
                                                norm_layer(16),
                                                nn.ReLU(True),)
                self.offset_fc2 = nn.Conv2d(16, input_channel,
                                            3, 1, 1)
                self.pool = nn.MaxPool2d(2, 2)
                # Init offset LocalizationNetwork
                self.init_modules(self.offset_fc1)
                self.init_modules(self.offset_fc2)

            self.init_modules(self.sp_net)
            self.init_modules(self.spt_convnet)
            self.init_modules(self.stucture_fc1)
            if rand:
                nn.init.kaiming_normal_(self.stucture_fc2.weight.data, nonlinearity="relu")
                self.stucture_fc2.bias.data.fill_(0)
            else:
                self.init_spin(self.stucture_fc2, default_type*2)

    def init_modules(self, module):
        """
        Args:
            module (nn.Module): the modules to initialize
        """
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

    def init_spin(self, spin_fc2, nz):
        """
        Args:
            spin_fc2 (nn.Module): the 2nd fc layer in SPN, as Block8 in Tab.1 in Paper [1]
            nz (int): number of paired \betas exponents, which means the value of K x 2

        """
        init_id = [0.00]*nz+[5.00]
        if self.offsets:
            init_id += [-5.00]
            # init_id *=3
        init = np.array(init_id)
        spin_fc2.weight.data.fill_(0)

        if self.stn:
            F = self.F
            ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
            ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
            ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
            ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
            ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
            initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
            initial_bias = initial_bias.reshape(-1)
            init = np.concatenate([init, initial_bias], axis=0)
        spin_fc2.bias.data = torch.from_numpy(init).float().view(-1)

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): save path of pretrained model

        Returns:

        """
        return

    def forward(self, x, return_weight=False):
        """
        Args:
            x (torch.cuda.FloatTensor): input image batch
            return_weight (bool): set to False by default,
                                  if set to True return the predicted offsets of AIN, denoted as x_{offsets}

        Returns:
            torch.Tensor: rectified image [batch_size x I_channel_num x I_height x I_width], the same as the input size
        """

        assert x.data.type() == self.inputDataType
        if self.spt:
            feat = self.spt_convnet(x)
            fc1 = self.stucture_fc1(feat)
            sp_weight_fusion = self.stucture_fc2(fc1)
            sp_weight_fusion = sp_weight_fusion.view(x.size(0), self.out_weight, 1)
            if self.offsets:  # SPIN w. AIN
                lambda_color = sp_weight_fusion[:, self.spt_length, 0]
                lambda_color = self.sigmoid(lambda_color).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                sp_weight = sp_weight_fusion[:, :self.spt_length, :]
                offsets = self.pool(self.offset_fc2(self.offset_fc1(feat)))

                assert offsets.size(2) == 2  # 2
                assert offsets.size(3) == 6  # 16
                offsets = self.sigmoid(offsets)  # v12

                if return_weight:
                    return offsets
                offsets = nn.functional.upsample(offsets, size=(x.size(2), x.size(3)), mode='bilinear')

                if self.stn is not None:
                    batch_C_prime = sp_weight_fusion[:, (self.spt_length + 1):, :].view(x.size(0), self.F, 2)
                    build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
                    build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0),
                                                                   self.I_r_size[0],
                                                                   self.I_r_size[1],
                                                                   2])

            else:  # SPIN w.o. AIN
                sp_weight = sp_weight_fusion[:, :self.spt_length, :]
                lambda_color, offsets = None, None

                if self.stn is not None:
                    batch_C_prime = sp_weight_fusion[:, self.spt_length:, :].view(x.size(0), self.F, 2)
                    build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
                    build_P_prime_reshape = build_P_prime.reshape([build_P_prime.size(0),
                                                                   self.I_r_size[0],
                                                                   self.I_r_size[1],
                                                                   2])

            x = self.sp_net(x, sp_weight, offsets, lambda_color)
            if self.stn:
                x = F.grid_sample(x, build_P_prime_reshape, padding_mode='border')
        return x
