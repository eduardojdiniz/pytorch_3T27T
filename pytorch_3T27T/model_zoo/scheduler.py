#!/usr/bin/env python
# coding=utf-8

from pytorch_3T27T.base.scheduler import lr_scheduler, BaseScheduler


__all__ = ['StepLR']


def StepLR(optimizer, **defaults):
    return lr_scheduler.StepLR(optimizer, **defaults)
