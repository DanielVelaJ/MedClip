#!/usr/bin/env python3 
"""
A script to download datasets for medical image captioning. To run, go to the 
:file:`src` folder and run the following line. 

.. code-block:: console

   $ python download_data.py
   
The script will prompt for user input to know which datasets to download. 
A description of the available datasets can be found :doc:`here <datasets>` .

This script uses the functions in the :py:mod:`download module <download>` .


"""
if __name__ == '__main__':
    import download as ft
    import os
    import subprocess
    
    # Create the raw directory
    path = '../data/raw' 
    os.makedirs(path, exist_ok=True)
    print('created data/raw folders')
    
    # Prompt user to know which datasets to donwload. 
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