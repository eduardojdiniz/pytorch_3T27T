#!/usr/bin/env python
# coding=utf-8

from abc import ABC, abstractmethod
import numpy as np


__all__ = ['BaseModel']


class BaseModel(ABC):
    """Base class for all models"""

    @abstractmethod
    def __init__(self):
        """Init"""

    @abstractmethod
    def forward(self, x):
        """
        Forward pass logic

        Parameters
        ----------
        x : torch.Tensor
            Model input

        Returns
        -------
        Model output
        """

    def __str__(self):
        """
        Print the total number of trainable parameters in the model and (if
        verbose) model architecture
        """
        print('----- Networks initialized -----')
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params}"
