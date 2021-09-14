PyTorch 3T27T
==============================

3T27T: Image-to-image translation of 3T to 7T brain MRI using Generative Adversarial Networks - a step towards longitudinal harmonization

Project Organization
------------
```
├── LICENSE
├── Makefile             <- Makefile with commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
│
├── references/          <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures/         <- Generated graphics and figures to be used in reporting
│
├── docs/                <- A default Sphinx project; see sphinx-doc.org for details
│
├── data                 <- directory for storing input data
│   ├── external/        <- Data from third party sources.
│   ├── interim/         <- Intermediate data that has been transformed.
│   ├── processed/       <- The final, canonical data sets for modeling.
│   └── raw/             <- The original, immutable data dump.
│
├── experiments                                     <- directory for storing experimental setups
│   ├── A01_MLP-XOR-implementation/
│   ├── A02_clustering-interleaving-moons/
│   ├── A03_MNIST-handwritten-digit-classification/
│   └── A04_fashion-MNIST-clothing-classification   <- Aims. Naming convention is
│           │                                          aim ID (A + ordered two-digit integer)
│           │                                          + "_" + short "-" delimited description
│           ├── E01_linear-model/
│           ├── E02_MLP/
│           └── E03_Alpha                              <- Experiments. Naming convention is
│                │                                     experiment ID
│                │                                     (E + ordered two-digit integer)
│                │                                     + "_" + short "-" delimited description
│                ├── S0001_pilot/
│                ├── S0002_dec-lr/
│                └── S0003_inc-batch                <- Setups. Naming convention is setup ID
│                      │                               (S + ordered four-digit integer)
│                      │                               + "_" + short "-" delimited description
│                      └── A04-E03-S0003.yml        <- YAML configuration file.
│                                                      Naming convention is aim ID +
│                                                      "-" + experiment ID + "-" setup ID
│
│
├── notebooks/                     <- Jupyter notebooks. Naming convention is
│                                     a number (for ordering) + "_" + the creator's initials
│                                     + "_" experiment ID + "_" + short `-` delimited
│                                     description, e.g. `1.0_ed_E01_results-visualization`
│
├── trials                         <- directory for storing experimental trials
│    ├── A01-E01-S0001/            <- Name of the YAML config file
│    ├── A02-E01-S0001/
│    ├── A03-E01-S0001/
│    └── A04-E03-S0003
│         ├── timestamp1/           <- timestamp, in YYYYMMDDhhmmssUTC format,
│         │                            i.e., year (YYYY), month (MM), day (DD),
│         │                            hour (hh), minute (mm), second (ss), all according to
│         │                            the Coordinated Universtal Time (UTC) standard
│         └── timestamp2
│                    ├── etc/       <- stores all information necessary to reproduce this trial,
│                    │                 e.g., the YAML config file, resuming session, seeds
│                    ├── logs/      <- stores all logging information
│                    ├── runs/      <- stores Tensorboard runs
│                    └── ckpts/     <- stores checkpoints
│
│
├── observations                    <- directory for storing experimental observations
│    ├── A01-E01-S0001/
│    ├── A02-E01-S0001/
│    ├── A03-E01-S0001/
│    └── A04-E03-S0003
│         ├── timestamp1/
│         └── timestamp2/
│
│
├── tox.ini              <- tox file with settings for running tox; see tox.readthedocs.io
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment,
│                           e.g., generated with `pip freeze > requirements.txt`
├── requirements-dev.txt <- The requirements file with the devel dependencies
│
├── setup.py             <- makes project pip installable (pip install -e .)
│                           so package_name can be imported
├── tests/               <- folder with test code
│
├── logging.yml          <- logging configuration
│
└── package_name                  <- package directory
    ├── __init__.py               <- Makes package_name a Python package
    ├── cli.py                    <- command line interface
    ├── package_name.py           <- package script
    ├── main_foo.py               <- training script for model "foo"
    ├── main_alpha.py             <- training script for model "alpha"
    │
    ├── base                      <- abstract base classes
    │   ├── __init__.py           <- Makes the abstract base class a Python subpackage
    │   ├── transform.py          <- abstract base class for data transformations
    │   ├── dataset.py            <- abstract base class for datasets
    │   ├── dataloader.py         <- abstract base class for data loaders
    │   ├── arch.py               <- abstract base class for models' archtectures
    │   ├── loss.py               <- abstract base class for losses
    │   ├── metric.py             <- abstract base class for metrics
    │   ├── optimizer.py          <- abstract base class for optimizers
    │   ├── scheduler.py          <- abstract base class for schedulers
    │   └── trainer.py            <- abstract base class for trainers
    │
    ├── utils                     <- utilities
    │    ├── seeder.py            <- manages reproducibility
    │    ├── logger.py            <- trainning logger
    │    ├── monitor.py           <- Tensorboard visualization support
    │    └── backer.py            <- manages paths for saving models + logs
    │
    ├── model_foo/                <- each model is a subpackage
    └── model_alpha               <- model "alpha"
         ├── __init__.py          <- Makes model_alpha a Python subpackage
         ├── dataset.py
         ├── transform.py
         ├── dataloader.py
         ├── loss.py
         ├── metric.py
         ├── arch.py
         ├── optimizer.py
         ├── scheduler.py
         └── trainer.py

```

 <p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
<p><small>Some features inspired by <a target="_blank" href="https://github.com/uwescience/shablona">A template for small scientific python projects</a>. #shablona</small></p>
<p><small>Package scaffold inspired by <a target="_blank" href="https://github.com/khornlund/cookiecutter-pytorch">cookiecutter PyTorch Deep Learning template</a>. #cookiecutterpytorch</small></p>

### Config file format
------------------
Config files are in `.yml` format:

```yaml
name: A04-E03-S0003
seed: 12345
target_devices: [0]
save_dir: trials/

arch:
    type: Alpha
    args: {}

transform:
    type: MNISTTransform
    args: {}

dataloader:
    type: MNISTDataloader
    args:
        batch_size: 128
        data_dir: data/
        num_workers: 2
        shuffle: true
        validation_split: 0.1

loss: nll_loss

lr_scheduler:
    type: StepLR
    args:
        gamma: 0.1
        step_size: 50

metrics:
    - top_1_acc
    - top_3_acc

optimizer:
    type: Adam
    args:
        lr: 0.001
        weight_decay: 0

training:
    early_stop: 10
    epochs: 100
    monitor: min val_loss
    save_period: 1
    tensorboard: true

testing:
    data_dir: data/
    batch_size: 128
    num_workers: 8
```

Add additional configurations as needed.

### Test the newly create repo
------------

1. Create virtual environment
```bash
$ make virtual_environment
```

2. Instal development requirements
```bash
$ make install_requirements
```

3. Run MNIST hello world example
```bash
$ conda activate <your_repo_name>
$ <your_repo_name> train-hello
```

[![asciicast](https://asciinema.org/a/244658.svg)](https://asciinema.org/a/244658)

### Checkpoints
-----------
You can specify the name of the training session in config files:

```yaml
name: A04-E03-S0003
```

The checkpoints will be saved in

`<save_dir>/<trial>/<timestamp>/ckpts/ckpt_epoch-<n>_<trial>_<timestamp>`,

where `<save_dir>` and `<trial>` are as defined in the config file, `<n>` is an
integer identifying the checkpoint, and timestamp is a datetime footprint in
the `YYYYMMDDhhmmssUTC` format, with year (`YYYY`), month (`MM`), day (`DD`),
hour (`hh`), minute (`mm`), second (`ss`), all according to the Coordinated
Universtal Time (`UTC`) standard. A copy of config file will be saved in

`<save_dir>/<trial>/<timestamp>/etc/config_<trial>_<timestamp>.yml`.

**Note**: checkpoints contain:

```python
checkpoint = {
'arch': arch,
'epoch': epoch,
'state_dict': self.model.state_dict(),
'optimizer': self.optimizer.state_dict(),
'monitor_best': self.mnt_best,
'config': self.config
}
```

### Tensorboard Visualization
--------------------------
This template supports [Tensorboard visualization](https://pytorch.org/docs/stable/tensorboard.html)

1. Run training

> Set `tensorboard` option in config file true.

2. Open tensorboard server

> Type `tensorboard --logdir <save_dir>/` at the project root, then server will
> open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, input images,
and histogram of model parameters will be logged. If you need more
visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc
in the `trainer._train_epoch` method. `add_something()` methods in this
template are basically wrappers for those of `tensorboard.SummaryWriter`
module.

**Note**: You don't have to specify current steps, since `TensorboardWriter`
class defined at `logger/monitor.py` will track current steps automatically.
