"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    __init__.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2021-09-18
##################################################################################################
"""

from .metric import TEDS
from .format import format_html
from .parallel import parallel_process

__all__ = ['TEDS', 'format_html', 'parallel_process']
