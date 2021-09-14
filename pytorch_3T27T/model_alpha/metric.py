#!/usr/bin/env python
# coding=utf-8

import torch
from pytorch_3T27T.base.metric import top_k_acc


__all__ = ['top_1_acc', 'top_3_acc']


def top_1_acc(output, target):
    """
    Computes the top 1 accuracy
    """
    return top_k_acc(output, target, k=1)


def top_3_acc(output, target):
    """
    Computes the top 3 accuracy
    """
    return top_k_acc(output, target, k=3)
