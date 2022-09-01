# -*- coding: utf-8 -*-
"""
This module contains functions to download the datasets for medclip project. 

All of the functions in this module are to be called from the download_data script. 
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


__version__ = '0.1'
__author__ = 'Daniel Vela Jarquin'
__email__ = 'd.vela1@hotmail.com'
__project__ = 'MedClip'

raw_data_path = '../data/raw/'

#print("fetch called from \n"+os.getcwd())


# Streaming, so we can iterate over the response.


def extract_from_url(url=' https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-large-zip-file.zip', path='here'):
    '''
    This function takes a zip download url and extracts it to the given path.

    The function also graphs a progress bar that indicates download speeds.

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
    '''
    Download medpix (not working for now, see function definition file comments).

    This function downloads the medpix dataset from Mindkind's google
    drive folder:
    https://drive.google.com/file/d/1-IVgCEGSzPc8NExdWFu8mTTdnhlysU6b/view?usp=sharing
    '''
    # Code below commented because it stopped working after googledrive made
    # changes for now you can download medpix.zip and manually place it in the
    # raw folder before running the unzip function in this method. The file can
    # be found here under the name Production https://drive.google.com/drive/folders/1PyI-R7x9o4pzR-s6pkHb-7XVZWh58s7Z?usp=sharing
# =============================================================================
#     print('downloading medpix')
#     try:
#         gdd.download_file_from_google_drive(file_id='1-IVgCEGSzPc8NExdWFu8mTTdnhlysU6b',
#                                             dest_path=raw_data_path+'medpix.zip',
#                                             unzip=True)
#         print('renaming')
#         rename(raw_data_path+'Production', raw_data_path+'medpix')
#         remove(raw_data_path+'medpix.zip')
#         print('done')
#     except Exception as e:
#         print('Error downloading medpix')
#         print(e)
# =============================================================================
    print('downloading medpix')
    def dropbox_connect():
        """Create a connection to Dropbox."""

        try:
            dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
        except AuthError as e:
            print('Error connecting to Dropbox with access token: ' + str(e))
        return dbx

    def dropbox_download_file(dropbox_file_path, local_file_path):
        """Download a file from Dropbox to the local machine."""

        try:
            dbx = dropbox_connect()
            print('Connection succesfull, starting download...')

            with open(local_file_path, 'wb') as f:
                metadata, result = dbx.files_download(path=dropbox_file_path)
                f.write(result.content)
        except Exception as e:
            print('Error downloading file from Dropbox: try shutting down all kernels and retrying it may be memory error. ')
            
        
    # Download zip from dropbox
    DROPBOX_ACCESS_TOKEN = input("Enter Daniel's dropbox access token:\n")
    dropbox_file_path = '/Global_datasets/medpix.zip'
    zip_path = '../data/raw/medpix.zip'
    dropbox_download_file(dropbox_file_path, zip_path)

    path = '../data/raw/'
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    remove(zip_path)
    rename(raw_data_path+'Production', raw_data_path+'medpix')


def chexpert():
    '''
    This function downloads the chexpert dataset (small version)
    from the url provided to us when we made stanford signin.
    '''
    print('downloading chexpert')
    chexpert_url = ('https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload'
                    '.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0-small.zip&h=7af997182d92'
                    '98f2fdbc10da45f4f70341158f140e891ea3e6ef48d0e7d65646&v=1&xid=eed9'
                    '8b1fbb&uid=55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+S'
                    'ubscription+Confirmed')
    extract_from_url(chexpert_url, raw_data_path + 'chexpert')
# =============================================================================
#         chexpert_url ='https://sample-videos.com/zip/20mb.zip'
# =============================================================================
