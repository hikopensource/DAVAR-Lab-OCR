"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-05-01
##################################################################################################
"""
from .backbones import *
from .losses import *
from .sequence_heads import *
from .transformations import *
from .recognizors import *
from .connects import *

from .builder import RECOGNIZORS, build_recognizor

