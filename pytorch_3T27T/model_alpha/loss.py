#!/usr/bin/env python
# coding=utf-8

import torch.nn.functional as F
from pytorch_3T27T.base import BaseLoss


__all__ = ['nll_loss']


def nll_loss(output, target):
    return F.nll_loss(output, target)
