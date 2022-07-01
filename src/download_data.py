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


# ft.medpix()
ft.chexpert()