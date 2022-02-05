# -*- coding: utf-8 -*-
"""
This script reads the metadata from the raw folders for each dataset and 
generates clean dataframes. The clean df are stored in the data/intermediate
directory

"""

import preprocess.prepare as prepare
prepare.medclip()
prepare.chexpert()