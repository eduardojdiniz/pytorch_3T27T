#!/usr/bin/env python
# coding=utf-8

# For PyTorch custom loss functions see:
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch/comments

from abc import ABCMeta, abstractmethod
import torch.nn as nn


__all__ = ['BaseLoss']


class BaseLoss(nn.Module, metaclass=ABCMeta):
    """Base class for all custom losses"""

    def __init__(self, weight=None, reduction='mean'):
        """
        Parameters
        ----------
        weight : torch.Tensor (optional)
            A manual rescaling weight given to the loss of each batch element.
            If given, has to be a Tensor of size nbatch.
        reduction : str (optional)
            Specifies the reduction to apply to the output: 'none' | 'mean' |
            'sum'. 'none': no reduction will be applied, 'mean': the sum of the
            output will be divided by the number of elements in the output,
            'sum': the output will be summed. Note: size_average and reduce are
            in the process of being deprecated, and in the meantime, specifying
            either of those two args will override reduction. Default: 'mean'
        """

        super().__init__()
        self.weight = weight
        self.reduction = reduction

    @abstractmethod
    def forward(self, inputs, targets, **kwargs):
        """Compute the loss"""
