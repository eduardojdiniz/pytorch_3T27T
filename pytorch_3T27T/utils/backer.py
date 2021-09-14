#!/usr/bin/env python
# coding=utf-8

# Manages Paths for Saving Models and Logs

from pathlib import Path
from os.path import join as pjoin

import pytorch_3T27T as pkg
root = Path(pkg.__path__[0]).parent.absolute()

ETC_DIR = "etc"
LOG_DIR = "logs"
CHECKPOINT_DIR = "ckpts"
RUN_DIR = "runs"


def ensure_exists(p: Path) -> Path:
    """
    Helper to ensure a directory exists.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def trial_path(config: dict) -> Path:
    """
    Construct a path based on the name of a configuration file,
    e.g. 'trials/A01-E01-S0001'
    """
    p = pjoin(root, config["save_dir"], config["trial_info"]["ID"])
    return ensure_exists(p)


def trial_timestamp_path(config: dict) -> Path:
    """
    Construct a path based on the name of a configuration file and append a
    timestamp, e.g. 'trials/A01-E01-S0001/20211231235959UTC'
    """
    timestamp = config["trial_info"]["timestamp"]
    p = pjoin(trial_path(config), timestamp)
    return ensure_exists(p)


def etc_path(config: dict) -> Path:
    """
    Retuns the config dir, e.g. 'trials/A01-E01-S0001/20211231235959UTC/etc'
    """
    p = pjoin(trial_timestamp_path(config), ETC_DIR)
    return ensure_exists(p)


def log_path(config: dict) -> Path:
    """
    Retuns the log dir, e.g. 'trials/A01-E01-S0001/20211231235959UTC/logs'
    """
    p = pjoin(trial_timestamp_path(config), LOG_DIR)
    return ensure_exists(p)


def trainer_paths(config: dict) -> Path:
    """
    Returns the paths to save checkpoints and tensorboard runs, e.g.

        trials/A01-E01-S0001/20211231235959UTC/ckpts
        trials/A01-E01-S0001/20211231235959UTC/runs
    """
    trial_timestamp = trial_timestamp_path(config)
    return (
        ensure_exists(pjoin(trial_timestamp, CHECKPOINT_DIR)),
        ensure_exists(pjoin(trial_timestamp, RUN_DIR)),
    )
