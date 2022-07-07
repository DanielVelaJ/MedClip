#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
This script reads the metadata from the raw folders for each dataset and
generates clean dataframes. The clean df are stored in the data/intermediate
directory as .csv files.

"""

import preprocess.prepare as prepare
import os

option= input('Prepare chexpert? (enter option [y] , [n]):\n')
if (option=='y'):
    prepare.chexpert()
    
option= input('Prepare medpix? (enter option [y] , [n]):\n')
if (option=='y'):   
    prepare.medpix()

option= input('Prepare mimic? (enter option [y] , [n]):\n')
if (option=='y'):   
    prepare.mimic()