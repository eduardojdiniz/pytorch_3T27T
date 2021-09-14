#!/usr/bin/env python
# coding=utf-8

from __future__ import absolute_import, division, print_function

from .due import due, Doi, BibTeX, Text

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("00.0000/00.0.00"),
         description="3T27T: Image-to-image translation of 3T to 7T brain MRI using Generative Adversarial Networks - a step towards longitudinal harmonization",
         tags=["[OPTIONAL] tags or keywords relevant to the project"],
         path='pytorch_3T27T')
