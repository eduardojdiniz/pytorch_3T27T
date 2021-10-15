#!/usr/bin/env python
# coding=utf-8

# Trainning Logger

import yaml
import logging
import logging.config
from pathlib import Path

from .backer import get_log_path

# When you set a logging level in Python using the standard module, you’re
# telling the library you want to handle all events from that level on up. If
# you set the log level to INFO, it will include INFO, WARNING, ERROR, and
# CRITICAL messages. NOTSET and DEBUG messages will not be included here.

# Logging levels list

# Level	    Numeric value
#  CRITICAL   50
#  ERROR      40
#  WARNING    30
#  INFO	      20
#  DEBUG      10
#  NOTSET     00
LOG_LEVEL = logging.INFO


__all__ = ['setup_logging', 'setup_logger']


def setup_logging(trial_config, log_config="logging.yml") -> None:
    """
    Setup ``logging.config``

    Parameters
    ----------
    trial_config : str
        Path to configuration file for run

    log_config : str
        Path to configuration file for logging
    """
    log_config = Path(log_config)

    if not log_config.exists():
        logging.basicConfig(level=LOG_LEVEL)
        logger = logging.getLogger("setup")
        msg = f'"{log_config}" not found. Using basicConfig.'
        logger.warning(msg)

    with open(log_config, "rt") as f:
        config = yaml.safe_load(f.read())

    # modify logging paths based on run config
    trial_path = get_log_path(trial_config)
    for _, handler in config["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(trial_path / handler["filename"])

    logging.config.dictConfig(config)


def setup_logger(name):
    log = logging.getLogger(f'pytorch_3T27T.{name}')
    log.setLevel(LOG_LEVEL)
    return log
