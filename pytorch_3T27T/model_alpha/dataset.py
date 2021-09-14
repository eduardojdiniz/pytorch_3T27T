#!/usr/bin/env python
# coding=utf-8

from torchvision.datasets import MNIST
from pytorch_3T27T.base import BaseDataset


__all__ = ['MNISTDataset']


def MNISTDataset(data_dir, train=True, transform=None, target_transform=None,
                 download=False):
    return MNIST(data_dir, train, transform, target_transform, download)
