#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import, division, print_function

import os
import random
from typing import Any, List, Tuple, Dict
from types import ModuleType

import torch
import torch.nn as nn

from pytorch_3T27T.utils import set_seed
from pytorch_3T27T.utils import seed_worker, seed_generator

from pytorch_3T27T.utils import setup_logger

import pytorch_3T27T.model_alpha.transform as ModelTransform
import pytorch_3T27T.model_alpha.dataloader as ModelDataLoader
import pytorch_3T27T.model_alpha.arch as ModelArch
import pytorch_3T27T.model_alpha.loss as ModelLoss
import pytorch_3T27T.model_alpha.metric as ModelMetric
import pytorch_3T27T.model_alpha.optimizer as ModelOptimizer
import pytorch_3T27T.model_alpha.scheduler as ModelScheduler
import pytorch_3T27T.model_alpha.trainer as ModelTrainer


logger = setup_logger(__name__)


def get_instance(module: ModuleType, name: str,
                 config: Dict, *args: Any) -> Any:
    """
    Helper to construct an instance of a class.

    Parameters
    ----------
    module : ModuleType
        Module containing the class to construct.
    name : str
        Name of class, as would be returned by ``.__class__.__name__``.
    config : dict
        Dictionary containing an 'args' item, which will be used as ``kwargs``
        to construct the class instance.
    args : Any
        Positional arguments to be given before ``kwargs`` in ``config``.
    """
    ctor_name = config[name]['type']
    logger.info(f'Building: {module.__name__}.{ctor_name}')
    return getattr(module, ctor_name)(*args, **config[name]['args'])


def setup_device(model: nn.Module,
                 target_devices: List[int]) -> Tuple[torch.device, List[int]]:
    """
    Setup device: GPU if available, else CPU.

    Parameters
    ----------
    target_devices : List[int]
        list of device IDs
    logger : logging.Logger

    Returns
    -------
    model : torch.device
        torch model into configured device
    device : List[int]
        A list of device pointers to be used
    """
    available_devices = list(range(torch.cuda.device_count()))

    if not available_devices:
        msg = ("There is no GPU available on this machine. "
               "Training will be performed on CPU.")
        logger.warning(msg)
        device = torch.device('cpu')
        model = model.to(device)
        return model, device

    if not target_devices:
        msg = ("No GPU selected. "
               "Training will be performed on CPU.")
        logger.info(msg)
        device = torch.device('cpu')
        model = model.to(device)
        return model, device

    max_target_gpu = max(target_devices)
    max_available_gpu = max(available_devices)

    if max_target_gpu > max_available_gpu:
        msg = (f"Configuration requestes GPU #{max_target_gpu} "
               "but only #{max_available_gpu} available. "
               "Check the configuration and try again.")
        logger.critical(msg)
        raise Exception(msg)

    msg = (f"Using devices {target_devices} of "
           "available devices {available_devices}")
    logger.info(msg)
    device = torch.device(f'cuda:{target_devices[0]}')
    if len(target_devices) > 1:
        model = nn.DataParallel(model, device_ids=target_devices).to(device)
    else:
        model = model.to(device)
    return model, device


def resume_checkpoint(resume_path, model, optimizer, config):
    """
    Resume from saved checkpoint.

    Parameters
    ----------
    resume_path : str
    model : torch.device
    optimizer : torch.optim
    config : Dict

    Returns
    -------
    model : torch.device
    optimizer : torch.optim
    epoch : int
    """
    if not resume_path:
        return model, optimizer, 0

    logger.info(f'Loading checkpoint: {resume_path}')
    ckpt = torch.load(resume_path)
    model.load_state_dict(ckpt['state_dict'])

    # load optimizer state from checkpoint only if optimizer type is the same
    if ckpt['config']['optimizer']['type'] != config['optimizer']['type']:
        msg = ("Warning: Optimizer type given in config file is different from"
               " that of checkpoint. Optimizer parameters not being resumed.")
        logger.warning(msg)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])

    logger.info(f'Checkpoint "{resume_path}" loaded')
    return model, optimizer, ckpt['epoch']


def get_optimizer_params(model: nn.Module, config: Dict) -> List:
    """
    Returns a Dict inside a list, whose keys are 'params', that holds
    model.parameters(), and the optimizer hyperparameters, e.g., for the Adam
    optimizer, `lr` and `weight_decay`.
    """
    return [{'params': model.parameters(), **config}]


def train(cfg: Dict, resume_path: str) -> None:
    """
    Main function. It uses a configuration dictionary to set the random
    generators' seeds, get model instance into configured device, get optimizer
    and lr_scheduler instances, get dataloader (with appropriated transforms)
    instance, do data-validation split, get loss and metrics function handles,
    and creates a Trainer object. Finally calls the trainer train function.

    Parameters
    ----------
    cfg : Dict
    resume_path : str
        if not None, is the path to a checkpoints folder
    """
    logger.debug(f'Training: {cfg}')

    if "seed" in cfg:
        seed = set_seed(logger, seed=cfg['seed'])
    else:
        seed = set_seed(logger)
    logger.debug(f'seed: {seed}')


    model = get_instance(ModelArch, 'arch', cfg)
    model, device = setup_device(model, cfg['target_devices'])

    optimizer_params = get_optimizer_params(model, cfg['optimizer'])
    optimizer = get_instance(ModelOptimizer, 'optimizer', cfg,
                             optimizer_params)
    lr_scheduler = get_instance(ModelScheduler, 'lr_scheduler',
                                cfg, optimizer)
    model, optimizer, start_epoch = resume_checkpoint(resume_path, model,
                                                      optimizer, cfg)

    transform = get_instance(ModelTransform, 'transform', cfg)

    seed_dataloader = cfg.get("seed_dataloader", False)
    if seed_dataloader is True:
            worker_init_fn = seed_worker
            logger.info("Dataloader seed has been set.")
            generator = seed_generator(torch.generator, logger)
            dataloader = get_instance(ModelDataLoader, 'dataloader', cfg,
                                    transform, worker_init_fn, generator)
    else:
        dataloader = get_instance(ModelDataLoader, 'dataloader', cfg,
                                  transform)

    val_dataloader = dataloader.split_validation()

    logger.info('Getting loss and metric function handles')
    loss = getattr(ModelLoss, cfg['loss'])
    metrics = [getattr(ModelMetric, metric) for metric in cfg['metrics']]

    logger.info('Initialising trainer')
    trainer = ModelTrainer.AlphaTrainer(model, loss, metrics, optimizer,
                                        start_epoch=start_epoch,
                                        config=cfg,
                                        device=device,
                                        dataloader=dataloader,
                                        val_dataloader=val_dataloader,
                                        lr_scheduler=lr_scheduler)

    trainer.train()
    logger.info('Finished!')
