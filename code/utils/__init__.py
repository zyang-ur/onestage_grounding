# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Edgar Andrés Margffoy-Tuay, Emilio Botero and Juan Camilo Pérez
#
# Licensed under the terms of the MIT License
# (see LICENSE for details)
# -----------------------------------------------------------------------------

"""Misc data and other helping utillites."""

from .word_utils import Corpus
from .transforms import ResizeImage, ResizeAnnotation

Corpus
ResizeImage
ResizeAnnotation


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
