#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import, division, print_function
import os.path
from os.path import join as pjoin
import pytorch_3T27T
from .due import due, Doi, BibTeX, Text

# Project root folder
SRC = os.path.dirname(pytorch_3T27T.__file__)
ROOT = os.path.dirname(SRC)
DATA_ROOT = pjoin(ROOT, 'datasets')

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("00.0000/00.0.00"),
         description=("p3T27T: Image-to-image translation of 3T to 7T ",
                      "brain MRI using Generative Adversarial Networks ",
                      "- a step towards longitudinal harmonization"),
         tags=["[OPTIONAL] tags or keywords relevant to the project"],
         path='pytorch_3T27T')
