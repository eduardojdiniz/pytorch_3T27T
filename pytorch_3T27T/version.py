from __future__ import absolute_import, division, print_function
from os.path import join as pjoin
from pathlib import Path
import io
import re
import pytorch_3T27T as pkg
root = Path(pkg.__path__[0]).parent.absolute()
readme = pjoin(root, 'README.md')

def read_long_description(readme):
    text_type = type(u"")
    with io.open(readme, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"),
                      fd.read())

# Format expected by setup.py and docs/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD-3-Clause",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Programming Language :: Python :: 3.8",
               "Topic :: Scientific/Engineering"]

NAME = "pytorch_3T27T"
MAINTAINER = "Eduardo Diniz"
MAINTAINER_EMAIL = "eduardojdiniz@gmail.com"
DESCRIPTION = ("p3T27T: Image-to-image translation of 3T to 7T brain MRI ",
               "using Generative Adversarial Networks ",
               "- a step towards longitudinal harmonization"),
LONG_DESCRIPTION = read_long_description(readme)
URL = "http://github.com/eduardojdiniz/pytorch_3T27T"
DOWNLOAD_URL = ""
LICENSE = "BSD-3-Clause"
AUTHOR = "Eduardo Diniz"
AUTHOR_EMAIL = "eduardojdiniz@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'pytorch_3T27T': [pjoin('data', '*')]}
REQUIRES = [
    "pytorch=1.9.0",
    "torchvision",
    "torchaudio",
    "cudatoolkit=11.1",
    "tensorboard",
    "numpy",
    "scipy",
    "scikit-learn",
    "scikit-image",
    "matplotlib",
    "seaborn",
    "pandas",
    "click",
    "absl-py",
    "pyyaml",
    "tqdm",
    "sphinx",
    "awscli",
    "python-dotenv",
    "duecredit",
    "watermark",
    "torchinfo",
    "dominate",
    "visdom",
    "Pillow",
    "ipywidgets",
] # use environment.yml for conda or requirements.txt for pip
PYTHON_REQUIRES = ">= 3.7"
