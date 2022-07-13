"""
##################################################################################################
# Copyright Info :    Copyright (c) Davar Lab @ Hikvision Research Institute. All rights reserved.
# Filename       :    davar_progressbar.py
# Abstract       :    Implementation of the progressbar of davar group.

# Current Version:    1.0.0
# Date           :    2022-05-18
##################################################################################################
"""

import sys
from collections.abc import Iterable
from multiprocessing import Pool
from shutil import get_terminal_size
import time

from mmcv.utils import Timer


class DavarProgressBar(object):
    """A progress bar which can print the progress"""

    def __init__(self, task_num=0, bar_width=50, start=True,
                 file=sys.stdout, min_time_interval=60):
        """
        Args:
            task_num (int): task number
            bar_width (int): the width of the progressbar in the terminal
            start (bool): start to record the time
            file (str): the info to save in the file
            min_time_interval (int): minimal the time interval to update the progressbar
        """
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()
        self.min_time_interval = min_time_interval
        self.print_time_last = time.time()

    @property
    def terminal_width(self):
        """

        Returns:
            int: terminal with
        """
        width, _ = get_terminal_size()
        return width

    def start(self):
        """
        start the timer
        """
        if self.task_num > 0:
            self.file.write('[{}] 0/{}, elapsed: 0s, ETA:'.format(
                ' ' * self.bar_width, self.task_num))
        else:
            self.file.write('completed: 0, elapsed: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self):
        """
        update the progressbar

        """
        self.completed += 1
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            if percentage <= 1:
                eta = int(elapsed * (1 - percentage) / percentage + 0.5)
                msg = '\r[{{}}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s' \
                      ''.format(self.completed, self.task_num, fps,
                                int(elapsed + 0.5), eta)

                bar_width = min(self.bar_width,
                                int(self.terminal_width - len(msg)) + 2,
                                int(self.terminal_width * 0.6))
                bar_width = max(2, bar_width)
                mark_width = int(bar_width * percentage)
                bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
                print_time_now = time.time()
                if print_time_now - self.print_time_last > self.min_time_interval:
                    self.file.write(msg.format(bar_chars))
                    self.print_time_last = print_time_now
        else:
            print_time_now = time.time()
            if print_time_now - self.print_time_last > self.min_time_interval:
                self.file.write(
                    'completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                        self.completed, int(elapsed + 0.5), fps))
                self.print_time_last = print_time_now
        self.file.flush()
