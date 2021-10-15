#!/usr/bin/env python
# coding=utf-8

from abc import ABC, abstractmethod
import torch.optim.lr_scheduler as lr_scheduler


__all__ = ['BaseScheduler', 'lr_scheduler']


class BaseScheduler(ABC):
    """Base class for all custom learning rate schedulers"""

    @abstractmethod
    def __init__(self):
        """Init"""
