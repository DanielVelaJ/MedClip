# -*- coding: utf-8 -*-
"""
This script reads the metadata from the raw folders for each dataset and
generates clean dataframes. The clean df are stored in the data/intermediate
directory as .csv files.

"""

import preprocess.prepare as prepare
import os

os.chdir('C:/Users/danic/MedClip/src')
prepare.chexpert()
