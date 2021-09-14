#!/usr/bin/env python
# coding=utf-8

from torch.utils.data import Dataset


__all__ = ['BaseDataset']


class BaseDataset(Dataset):
    """
    Base class for all custom datasets
    """
    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
