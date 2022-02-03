# -*- coding: utf-8 -*-
"""
This module contains functions to download the datasets
"""
from google_drive_downloader import GoogleDriveDownloader as gdd

import requests
import zipfile
import io



__version__ = '0.1'
__author__ = 'Daniel Vela Jarquin'
__email__ = 'd.vela1@hotmail.com'

raw_data_path='../data/raw/'

#print("fetch called from \n"+os.getcwd())
def medpix():
    '''This function downloads the medpix dataset from Mindkind's google 
    drive folder:
    https://drive.google.com/file/d/1P1rOTQup5zJNgD9k7mlwmejwocpF1ngl/
    view?usp=sharing  '''

    print('downloading medpix')
    try:
        gdd.download_file_from_google_drive(file_id='1P1rOTQup5zJNgD9k7mlwmejwocpF1ngl',
                                            dest_path=raw_data_path+'medpix.zip',
                                            unzip=False)
        print('done')
    except:
        print('errors downloading medclip')
    return


def chexpert():
    print('downloading chexpert')
    try:
        chexpert_url = ('https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload'
                        '.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0-small.zip&h=7af997182d92'
                        '98f2fdbc10da45f4f70341158f140e891ea3e6ef48d0e7d65646&v=1&xid=eed9'
                        '8b1fbb&uid=55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+S'
                        'ubscription+Confirmed')
        r = requests.get(chexpert_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.write(raw_data_path+'chexpert.zip')
        print('done')
    except:
        print('errors downloading chexpert')


