#!/usr/bin/env python
# coding=utf-8

# Manages Reproducibility

import os
import numpy as np
import random
import torch


def set_seed(logger, seed=None, seed_torch=True, seed_cudnn_benchmark=True,
             seed_cudnn_deterministic=True):
    """Set seed of random number generators to limit the number of sources of
    nondeterministic behavior for a specific platform, device, and PyTorch
    release. For more information, see
    https://pytorch.org/docs/stable/notes/randomness.html.

    Parameters
    ----------
    logger : logging.Logger object
    seed : int
        Seed for random number generators.
    seed_torch : bool
        If we will set the seed for torch random number generators.
        Default: True

    Returns
    -------
    seed : int
        Seed used for random number generators
    """
    if seed is None:
        seed_str = "pytorch_3T27T"
        seed = [str(ord(letter) - 96) for letter in seed_str]
        seed = abs(int(''.join(seed))) % 2 ** 32
        #seed = np.random.choice(2 ** 32)

    # Set python seed for custom operators
    random.seed(seed)

    # Set seed for the global NumPy RNG if any of the libraries rely on NumPy
    np.random.seed(seed)

    # Fix seed for generating the hash() of the types covered by the hash
    # randomization, i.e., str, bytes, and datetime objects
    os.environ['PYTHONHASHSEED'] = str(seed)

    if seed_torch:
        # Seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        if seed_cudnn_benchmark:
            # Set cuDNN to deterministically select a convolution algorithm
            # Don't set if not consistent input sizes
            torch.backends.cudnn.benchmark = True
        if seed_cudnn_deterministic:
            # Ensure that the cuDNN convolution algorithm is deterministic
            torch.backends.cudnn.deterministic = True

    logger.info(f'Random seed {seed} has been set.')

    return seed


def seed_worker(worker_id, logger):
    """Set seed for Dataloader. DataLoader will reseed workers the "Randomness
    in multi-process data loading" algorithm. Use `worker_init_fn()` to
    preserve reproducibility. For more information, see
    https://pytorch.org/docs/stable/notes/randomness.html.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_generator(generator, logger, seed=0):
    """Set seed for Dataloader generators. DataLoader will reseed workers the
    "Randomness in multi-process data loading" algorithm. Use `generator` to
    preserve reproducibility. For more information, see
    https://pytorch.org/docs/stable/notes/randomness.html.

    Parameters
    ----------
    generator : torch.Generator
    logger : logging.Logger object
    seed : int
        Seed for random number generators.
        Default: 0

    Returns
    -------
    generator : torch.Generator
        Seeded torch generator
    """
    generator.manual_seed(seed)

    msg = f"Dataloader generator with seed {seed} has been created."
    logger.info(msg)

    return generator
