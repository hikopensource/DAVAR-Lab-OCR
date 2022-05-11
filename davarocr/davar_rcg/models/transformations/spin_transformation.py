"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    spin_transformation.py
# Abstract       :    Implementations of the spin transformation

# Current Version:    1.0.0
# Date           :    2021-03-07
##################################################################################################
"""
import functools
import math

import torch
import torch.nn as nn

import numpy as np

from davarocr.davar_common.models.builder import TRANSFORMATIONS, build_transformation


class SP_TransformerNetwork(nn.Module):
    """
    Sturture-Preserving Transformation (SPT) as Equa. (2) in Ref. [1]


    Ref: [1] SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition. AAAI-2021.
    """

    def __init__(self, nc=1, default_type=5):
        """ Based on SPIN

        Args:
            nc (int): number of input channels (usually in 1 or 3)
            default_type (int): the complexity of transformation intensities (by default set to 6 as the paper)
        """
        super(SP_TransformerNetwork, self).__init__()
        self.power_list = self.cal_K(default_type)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.InstanceNorm2d(nc)

    def cal_K(self, k=5):
        """

        Args:
            k (int): the complexity of transformation intensities (by default set to 6 as the paper)

        Returns:
            List: the normalized intensity of each pixel in [0,1], denoted as \beta [1x(2K+1)]

        """
        from math import log
        x = []
        if k != 0:
            for i in range(1, k+1):
                lower = round(log(1-(0.5/(k+1))*i)/log((0.5/(k+1))*i), 2)
                upper = round(1/lower, 2)
                x.append(lower)
                x.append(upper)
        x.append(1.00)
        return x

    def forward(self, batch_I, weights, offsets, lambda_color=None):
        """

        Args:
            batch_I (torch.Tensor): batch of input images [batch_size x nc x I_height x I_width]
            weights:
            offsets: the predicted offset by AIN, a scalar
            lambda_color: the learnable update gate \alpha in Equa. (5) as
                          g(x) = (1 - \alpha) \odot x + \alpha \odot x_{offsets}

        Returns:
            torch.Tensor: transformed images by SPN as Equa. (4) in Ref. [1]
                        [batch_size x I_channel_num x I_r_height x I_r_width]

        """
        batch_I = (batch_I + 1) * 0.5
        if offsets is not None:
            batch_I = batch_I*(1-lambda_color) + offsets*lambda_color
        batch_weight_params = weights.unsqueeze(-1).unsqueeze(-1)
        batch_I_power = torch.stack([batch_I.pow(p) for p in self.power_list], dim=1)

        batch_weight_sum = (batch_I_power * batch_weight_params).sum(dim=1)
        batch_weight_sum = self.bn(batch_weight_sum)
        batch_weight_sum = self.sigmoid(batch_weight_sum)
        batch_weight_sum = batch_weight_sum * 2 - 1
        return batch_weight_sum


@TRANSFORMATIONS.register_module()
class SPIN_ColorTransformer(nn.Module):
    """ Rectification Network of SPIN.
    Differently from the common used STN,
    SPIN regards the transformation in color/intensity space.
     """
    '''
    Usage Example:
    transformation=dict(
         type='SPIN_ColorTransformer',
         input_channel = 3,
         offsets = False,
         stn = None,
    ),

    '''
    def __init__(self, input_channel=1,
                 inputDataType='torch.cuda.FloatTensor',
                 offsets=False,
                 norm_type='BN',
                 stn=None,
                 default_type=6,
                 rand=False):
        """

        Args:
            input_channel (int): channel of input features,
                                set it to 1 if the grayscale images and 3 if RGB input
            inputDataType (str): the type of input data,
                                only support 'torch.cuda.FloatTensor' this version
            offsets (bool): set it to False if use SPN w.o. AIN,
                            and set it to True if use SPIN (both with SPN and AIN)
            norm_type (str): the normalization type of the module,
                            set it to 'BN' by default, 'IN' optionally
            stn (dict): the pipline of SPIN and STN,
                        set it to False if only use SPIN
            default_type (int): the K chromatic space,
                                set it to 3/5/6 depend on the complexity of transformation intensities
            rand (bool): initialize type, set it to False by default

        """
        super().__init__()
        self.nc = input_channel
        self.inputDataType = inputDataType
        self.spt = True  # SPN is necessary in SPIN, so set the parameter to True
        self.offsets = offsets

        self.stn = build_transformation(stn)

        if norm_type == 'BN':
            norm_layer = functools.partial(nn.BatchNorm2d,
                                           affine=True,
                                           track_running_stats=True)
        elif norm_type == 'IN':
            norm_layer = functools.partial(nn.InstanceNorm2d,
                                           affine=False,
                                           track_running_stats=False)
        else:
            raise NotImplementedError('normalization layer '
                                      '[%s] is not found' % norm_type)

        if self.spt:
            self.sp_net = SP_TransformerNetwork(input_channel, default_type)
            self.spt_convnet = nn.Sequential(
                                  # 32*100
                                  nn.Conv2d(input_channel, 32,
                                            3, 1, 1,
                                            bias=False),
                                  norm_layer(32),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(kernel_size=2,
                                               stride=2),
                                  # 16*50
                                  nn.Conv2d(32, 64,
                                            3, 1, 1,
                                            bias=False),
                                  norm_layer(64),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(kernel_size=2,
                                               stride=2),
                                  # 8*25
                                  nn.Conv2d(64, 128,
                                            3, 1, 1,
                                            bias=False),
                                  norm_layer(128),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(kernel_size=2,
                                               stride=2),
                                  # 4*12
                                  # nn.Conv2d(128, 256, 3, 1, 1,
                                  #           bias=False),
                                  # norm_layer(256), nn.ReLU(True),
                                  # nn.MaxPool2d(kernel_size=2, stride=2),
                                  # nn.Conv2d(256, 256, 3, 1, 1,
                                  #           bias=False),
                                  # norm_layer(256), nn.ReLU(True),# 2*6
                                  # nn.MaxPool2d(kernel_size=2, stride=2),
                                  # nn.Conv2d(256, 512, 3, 1, 1,
                                  #           bias=False),
                                  # norm_layer(512), nn.ReLU(True),# 1*3
                                  # nn.AdaptiveAvgPool2d(1),
                                  # nn.Flatten(1, -1),
                                  # batch_size x 512
                                  # nn.Linear(512, 256),
                                  # nn.BatchNorm1d(256),
                                  # nn.ReLU(True)
            )
            print(self.spt_convnet)
            self.stucture_fc1 = nn.Sequential(
                                  nn.Conv2d(128, 256,
                                            3, 1, 1,
                                            bias=False),
                                  norm_layer(256),
                                  nn.ReLU(True),
                                  nn.MaxPool2d(kernel_size=2,
                                               stride=2),
                                  nn.Conv2d(256, 256,
                                            3, 1, 1,
                                            bias=False),
                                  norm_layer(256),
                                  nn.ReLU(True),  # 2*6
                                  nn.MaxPool2d(kernel_size=2,
                                               stride=2),
                                  nn.Conv2d(256, 512,
                                            3, 1, 1,
                                            bias=False),
                                  norm_layer(512),
                                  nn.ReLU(True),  # 1*3
                                  nn.AdaptiveAvgPool2d(1),
                                  nn.Flatten(1, -1),  # batch_size x 512
                                  nn.Linear(512, 256),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(True)
                                )
            self.out_weight = 2*default_type+1
            if offsets:
                self.out_weight += 1
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
                nn.init.kaiming_normal_(self.stucture_fc2.weight.data,
                                        nonlinearity="relu")
                self.stucture_fc2.bias.data.fill_(0)
            else:
                self.init_spin(self.stucture_fc2,
                               default_type*2)  # exp-spin or stn

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
        init = np.array(init_id)
        spin_fc2.weight.data.fill_(0)
        spin_fc2.bias.data = torch.from_numpy(init).float().view(-1)

    def init_weights(self, pretrained=None):
        """

        Args:
            pretrained (str): save path of pretrained model


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
            # if x.size(2) > 32:
            #    x_small = nn.functional.upsample(x,
            #        size=(self.targetH, self.targetW),
            #        mode='bilinear')
            # else:
            #    x_small = x
            x_small = x
            feat = self.spt_convnet(x_small)
            fc1 = self.stucture_fc1(feat)
            sp_weight_fusion = self.stucture_fc2(fc1)
            sp_weight_fusion = sp_weight_fusion.view(x.size(0),
                                                     self.out_weight,
                                                     1)
            if self.offsets:
                lambda_color = sp_weight_fusion[:, -1, 0]
                lambda_color = self.sigmoid(lambda_color).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                sp_weight = sp_weight_fusion[:, :-1, :]
                offsets = self.pool(
                    self.offset_fc2(self.offset_fc1(feat)))
                # assert offsets.size(2) == 2 #2
                # assert offsets.size(3) == 6 #6
                offsets = self.sigmoid(offsets)

                if return_weight:
                    return offsets
                offsets = nn.functional.upsample(offsets,
                                                 size=(x.size(2),
                                                       x.size(3)),
                                                 mode='bilinear')
            else:
                sp_weight = sp_weight_fusion
                lambda_color, offsets = None, None
            x = self.sp_net(x, sp_weight, offsets, lambda_color)

        if self.stn is not None:
            x = self.stn(x)

        return x
