#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""
This script takes the raw datasets inside :file:`MedClip/data/raw` and 
preprocesses them to create **intermediate datasets** in the 
:file:`MedClip/data/intermediate` directory.  

.. _intermediate_datasets:

.. admonition:: About **intermediate datasets**
   :class: note

   Intermediate datasets are 
   .csv files that must have the following columns for the captioning workflow to 
   work:

   * Path: This column should contain the paths to each image. Each row represents
     one image. The path must be relative to the src folder. 
   * Full_Caption: This column should contain the full caption sequence that will 
     be used during training. It must contain a `<start>` token at the begining of 
     the caption and an `<end>` token at the end. 
  
     Ex: :code:`"<start> this is a caption. <end>"`
     
Most of the functionality of this script depends and makes direct use of the 
:py:mod:`prepare module <prepare>` .
  
.. important::

   To run this script make sure that you have downloaded the raw dataset you 
   will use through the :py:mod:`download_data.py <download_data>` script.

To run the script go to the :file:`MedClip/src` directory and run: 

.. code-block:: console

  (venv) $ python prepare_data.py



"""
if __name__ == '__main__':
    import prepare as prepare
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