#!/usr/bin/env python3 
"""
Created on Thu Feb  3 15:57:19 2022

@author: danic
"""
import download.fetch as ft
import os

path = '../data/raw'
if not os.path.exists(path):
    os.makedirs(path)
    print('created data/raw folders')

option= input('Download medpix? (enter option [y] , [n]):\n')
if (option=='y'):
    ft.medpix()
option= input('Download chexpert? (enter option [y] , [n]):\n')
if (option=='y'):
    ft.chexpert()
print('Please run: wget -r -N -c -np --user danielvelaj --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/ to get mimic-cxr')
