#!/usr/bin/env python3 
"""
Created on Thu Feb  3 15:57:19 2022

@author: danic
"""
import download.fetch as ft
import os
import subprocess

path = '../data/raw'
if not os.path.exists(path):
    os.makedirs(path)
    print('created data/raw folders')
option1= input('Download medpix? (enter option [y] , [n]):\n')
option2= input('Download chexpert? (enter option [y] , [n]):\n')
option3= input('Download mimic? (enter option [y] , [n]):\n')
if (option1=='y'):
    ft.medpix()

if (option2=='y'):
    ft.chexpert()
# print('Please run: wget -r -N -c -np --user danielvelaj --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/ to get mimic-cxr')

if (option3=='y'): 
    subprocess.call("./download_mimic.sh",shell=True)