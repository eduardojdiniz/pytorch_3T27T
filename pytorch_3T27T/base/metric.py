#!/usr/bin/env python
# coding=utf-8

import torch


__all__ = ['top_k_acc']


def top_k_acc(output, target, k):
    """
    Compute the top k accuracy

    Parameters
    ----------
    output : model output
    target : label
    k : within-ranking range

    Returns
    -------
    Model Accuracy for the within-k class range
    """
    # Get the first k predictions by the model
    pred = torch.topk(output, k, dim=1)[1]
    assert pred.shape[0] == len(target)
    correct = 0
    for i in range(k):
        correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
