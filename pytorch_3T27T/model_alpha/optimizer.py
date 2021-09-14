#!/usr/bin/env python
# coding=utf-8

import torch.optim as optim
from pytorch_3T27T.base.optimizer import BaseOptimizer


__all__ = ['Adam']


def Adam(*params, **defaults):
    return optim.Adam(*params, **defaults)
