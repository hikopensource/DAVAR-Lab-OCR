"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    collect_env.py
# Abstract       :

# Current Version:    1.0.0
# Date           :    2020-05-31
##################################################################################################
"""
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import davarocr


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['DAVAROCR'] = davarocr.__version__ + '+' + get_git_hash()[:7]
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
