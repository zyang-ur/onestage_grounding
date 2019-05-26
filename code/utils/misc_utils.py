# -*- coding: utf-8 -*-

"""
Misc download and visualization helper functions and class wrappers.
"""

import sys
import time
import torch
from visdom import Visdom


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


class VisdomWrapper(Visdom):
    def __init__(self, *args, env=None, **kwargs):
        Visdom.__init__(self, *args, **kwargs)
        self.env = env
        self.plots = {}

    def init_line_plot(self, name,
                       X=torch.zeros((1,)).cpu(),
                       Y=torch.zeros((1,)).cpu(), **opts):
        self.plots[name] = self.line(X=X, Y=Y, env=self.env, opts=opts)

    def plot_line(self, name, **kwargs):
        self.line(win=self.plots[name], env=self.env, **kwargs)
