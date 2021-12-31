"""
####################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    convfc_bbox_head_w_gcn.py
# Abstract       :    insert gcn module into convfc bbox head.

# Current Version:    1.0.0
# Date           :    2021-12-13
######################################################################################################
"""
import copy
import torch.nn as nn

from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import ConvFCBBoxHead
from mmdet.models.builder import build_head


@HEADS.register_module()
class ConvFCBBoxHeadWGCN(ConvFCBBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 relation_module=None,
                 *args,
                 **kwargs):
        super().__init__(
            num_shared_convs=num_shared_convs,
            num_shared_fcs=num_shared_fcs,
            num_cls_convs=num_cls_convs,
            num_cls_fcs=num_cls_fcs,
            num_reg_convs=num_reg_convs,
            num_reg_fcs=num_reg_fcs,
            conv_out_channels=conv_out_channels,
            fc_out_channels=fc_out_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            *args,
            **kwargs)

        if relation_module is not None:
            self.relation_module = nn.ModuleList()
            for i in range(len(self.shared_fcs)):
                self.relation_module.append(build_head(copy.deepcopy(relation_module)))

    @property
    def with_relation_module(self):
        """

        Returns:
            bool: Determine the model with the relation_module or not
        """
        return hasattr(self, 'relation_module') and self.relation_module is not None

    def forward(self, x, rois, img_metas):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            idx = 0
            for fc in self.shared_fcs:
                if self.with_relation_module:
                    x = fc(x)
                    x = self.relu(self.relation_module[idx](x, rois, img_metas))
                    idx += 1
                else:
                    x = self.relu(fc(x))

        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHeadWGCN(ConvFCBBoxHeadWGCN):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHeadWGCN, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

@HEADS.register_module()
class Shared4Conv1FCBBoxHeadWGCN(ConvFCBBoxHeadWGCN):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHeadWGCN, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
