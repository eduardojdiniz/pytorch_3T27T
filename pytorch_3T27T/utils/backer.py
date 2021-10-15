#!/usr/bin/env python
# coding=utf-8

# Manages Paths for Saving Models and Logs

from pathlib import Path
from os.path import join as pjoin
import re
import datetime


import pytorch_3T27T as pkg
root = Path(pkg.__path__[0]).parent.absolute()

ETC_DIR = "etc"
LOG_DIR = "logs"
CHECKPOINT_DIR = "ckpts"
RUN_DIR = "runs"


__all__ = [
    'get_trial_path', 'get_etc_path', 'get_log_path', 'get_trainer_paths',
    'get_trial_dict', 'get_timestamp'
]


def ensure_exists(p: Path) -> Path:
    """
    Helper to ensure a directory exists.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_trial_path(config: dict) -> Path:
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
    p = pjoin(get_trial_path(config), timestamp)
    return ensure_exists(p)


def get_etc_path(config: dict) -> Path:
    """
    Retuns the config dir, e.g. 'trials/A01-E01-S0001/20211231235959UTC/etc'
    """
    p = pjoin(trial_timestamp_path(config), ETC_DIR)
    return ensure_exists(p)


def get_log_path(config: dict) -> Path:
    """
    Retuns the log dir, e.g. 'trials/A01-E01-S0001/20211231235959UTC/logs'
    """
    p = pjoin(trial_timestamp_path(config), LOG_DIR)
    return ensure_exists(p)


def get_trainer_paths(config: dict) -> Path:
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


def get_trial_dict(filename: str) -> dict:
    """
    Get trial information from file name

    Returns
    -------
    trial : dict
        Trial information: ID, Aim, Experiment, Setup
    """

    # Parse filename to get this particular experimental run info
    trial = {}

    nameRegex = re.compile(r'((A\d\d)-(E\d\d)-(S\d\d\d\d))')
    trial_ID, aim_ID, exp_ID, setup_ID = nameRegex.search(filename).groups()
    trial['ID'] = trial_ID

    aimRegex = re.compile(f'({aim_ID})_([\\w\\-]+)')
    trial['Aim'] = aimRegex.search(filename).groups()[0]

    expRegex = re.compile(f'({exp_ID})_([\\w\\-]+)')
    trial['Experiment'] = expRegex.search(filename).groups()[0]

    setupRegex = re.compile(f'({setup_ID})_([\\w\\-]+)')
    trial['Setup'] = setupRegex.search(filename).groups()[0]

    trial['timestamp'] = get_timestamp()
    return trial


def get_timestamp() -> str:
    """
    Get this experimental run timestamp, e.g., 20211231235959UTC'
    """
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return timestamp + 'UTC'
