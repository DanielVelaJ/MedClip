# -*- coding: utf-8 -*-
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

print('downloading medpix')
ft.medpix()
print('downloading chexpert')
ft.chexpert()
print('Please run: wget -r -N -c -np --user danielvelaj --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/ to get mimic-cxr')
