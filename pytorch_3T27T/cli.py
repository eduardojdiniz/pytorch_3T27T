import click
import os
from dotenv import find_dotenv, load_dotenv

from pytorch_3T27T import main_alpha
from pytorch_3T27T.utils import setup_logging, load_config


@click.group()
def cli():
    """
    CLI for pytorch_3T27T
    """
    pass


@cli.command()
@click.option('-c', '--config-filename', multiple=True,
              default=[
                       'experiments/A04_MNIST-handwritten-digit-classification/E03_CNN/S0004_inc-batch/A04-E03-S0004.yml'],
              help=(
                    'Path to training configuration file. If multiple are '
                    'provided, runs will be executed in order'))
@click.option('-e', '--env-variables', multiple=True,
              help=('Environment variables\' names. Multiple variables are '
                    'supported'))
@click.option('-r', '--resume', default=None, type=str,
              help='path to checkpoint')
def train_hello(config_filename, env_variables, resume):
    """
    Entry point to start training run(s) for model `alpha`.
    """
    if len(config_filename) == 0:
        print("Please provide the path to a experimental setup configuration")
        print("file using the -c (--config-filename) flag")
        return
    configs = [load_config(f) for f in config_filename]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    env_variables = {e: os.environ.get(e) for e in env_variables}

    for config in configs:
        setup_logging(config)
        config['ENV'] = env_variables
        if resume is not None:
            config['resume_checkpoint'] = resume
        main_alpha.train(config, resume)
