"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from .davar_common import *
from .davar_det import *
from .davar_rcg import *
from .davar_spotting import *
from .davar_ie import *
from .davar_layout import *
from .davar_videotext import *
from .davar_table import *
from .davar_nlp_common import *
from .davar_ner import *
from .davar_order import *
from .davar_distill import *
from .mmcv import *
from .version import __version__

__all__ = ['__version__']
