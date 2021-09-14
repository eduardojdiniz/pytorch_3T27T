#!/usr/bin/env python
# coding=utf-8

# For an initial guide on PyTorch loss functions see:
# https://neptune.ai/blog/pytorch-loss-functions

# For advanced PyTorch custom loss functions see:
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch/comments

import abc
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['BaseLoss']


class BaseLoss(ABC):
    """
    Base class for all custom losses
    """
    def __init__(self):
        pass
