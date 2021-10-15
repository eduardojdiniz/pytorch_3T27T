#!/usr/bin/env python
# coding=utf-8

# For a guide on writing custon PyTorch optimizers see:
# http://mcneela.github.io/machine_learning/2019/09/03/Writing-Your-Own-Optimizers-In-Pytorch.html

from abc import ABCMeta, abstractmethod
from torch.optim import Optimizer


__all__ = ['BaseOptimizer']


class BaseOptimizer(Optimizer, metaclass=ABCMeta):
    """Base class for all custom optimizers"""

    @abstractmethod
    def __init__(self, params, defaults):
        super(BaseOptimizer, self).__init__(params, defaults)

    @abstractmethod
    def step(self, closure=None):
        """Step"""
