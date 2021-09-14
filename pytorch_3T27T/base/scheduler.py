#!/usr/bin/env python
# coding=utf-8

import abc
from abc import ABC

import torch.optim.lr_scheduler as lr_scheduler


__all__ = ['BaseScheduler', 'lr_scheduler']


class BaseScheduler(ABC):
    """
    Base class for all custom learning rate schedulers
    """
    def __init__(self):
        pass
