# -*- coding: utf-8 -*-
"""
This module contains functions to download the datasets for the MedClip project.
Check the available datasets :doc:`here <datasets>` .

"""
from google_drive_downloader import GoogleDriveDownloader as gdd
import requests
import zipfile
import io
from os import rename
from os import remove
from tqdm import tqdm
import pathlib
import pandas as pd
import dropbox
from dropbox.exceptions import AuthError
import subprocess


__version__ = '0.1'
__author__ = 'Daniel Vela Jarquin'
__email__ = 'd.vela1@hotmail.com'
__project__ = 'MedClip'

raw_data_path = '../data/raw/'

# print("fetch called from \n"+os.getcwd())


# Streaming, so we can iterate over the response.


def extract_from_url(url, path):
    '''Takes a zip download url and extracts it to the given path.
    

    Args: 
        url(str): The url to the .zip file.
        path(str): The path where the url will be saved.
        

    '''
    zip_path = path+'.zip'
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(zip_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    print('Extracting from zip')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    remove(zip_path)


def medpix():
    '''Download the medpix dataset into the data/raw/medpix directory. 
    
    This function will connect to Daniel's dropbox to download the raw medpix
    files. The dataset will be stored in the :file:`MedClip/data/raw/medpix` 
    directory. 

    '''
    extract_from_url('https://www.dropbox.com/s/g2l3f8rvpw2p81a/mepix.zip?dl=1',
                     os.path.join(raw_data_path,'medpix'))

    print('Downloading the medpix dataset', raw_data_path+'medpix')
    


def chexpert():
    '''This function downloads the chexpert dataset (small version).
    
    '''
    print('downloading chexpert')
    # url provided to us when we made stanford signin.
    chexpert_url = ('https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload'
                    '.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0-small.zip&h=7af997182d92'
                    '98f2fdbc10da45f4f70341158f140e891ea3e6ef48d0e7d65646&v=1&xid=eed9'
                    '8b1fbb&uid=55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+S'
                    'ubscription+Confirmed')
    extract_from_url(chexpert_url, raw_data_path + 'chexpert')
    
def mimic_cxr(): 
    """Download the mimic-cxr dataset using Daniel's physionet account. 
    
    This function requires the :file:`download_mimic.sh` bash file. It executes it. 
    """
    print('downloading mimic_cxr')
    subprocess.call("./download_mimic.sh",shell=True)
# =============================================================================
#         chexpert_url ='https://sample-videos.com/zip/20mb.zip'
# =============================================================================
