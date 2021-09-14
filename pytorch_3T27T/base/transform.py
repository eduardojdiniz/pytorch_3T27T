#!/usr/bin/env python
# coding=utf-8

import abc
from abc import ABC

import torchvision.transforms as T


__all__ = ['AugmentationFactory', 'BaseTransform']


class AugmentationFactory(ABC):
    """
    Base class for augmentation factory
    """
    def get_transform(self, train):
        """
        Switch to select between train or test transforms
        """
        return self.get_train() if train else self.get_test()

    @abc.abstractmethod
    def get_train(self):
        """
        Get transform to be applied to the train dataset
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_test(self):
        """
        Get transform to be applied to the test dataset
        """
        raise NotImplementedError


class BaseTransform():
    """
    Base class for all custom transforms
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        raise NotImplementedError
