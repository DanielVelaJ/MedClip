# -*- coding: utf-8 -*-
"""
This includes functions to make clean dataframes from zipfiles.

This module includes functions to extract the zip files for each dataset
after downloaded and to prepare dataframes containing clean and
 relevant information for each dataset. The resulting dataframes are
 saved as "intermediate" data in C:/Users/danic/MedClip/data/intermediate
 to facilitate merging.

 The format
"""
#TODO
# Fix the time estimates in the fix images function
# Add image checking to chexpert and to mimic

import pandas as pd
import random
import os
import cv2
import imghdr
import time
import numpy as np
import datetime
import pandas as pd
from PIL import Image
import datetime


def chexpert():
    '''
    Generates a clean dataframe from chexpert raw data stored in the raw/data
    directory.

    '''
    # Define paths for reading data
    chexpert_raw_path = '../data/raw/chexpert'
    path_train = chexpert_raw_path + '/CheXpert-v1.0-small/train.csv'
    path_valid = chexpert_raw_path + '/CheXpert-v1.0-small/valid.csv'
    save_path = '../data/intermediate/inter_chexpert.csv'

    df_train = pd.read_csv(path_train)
    df_valid = pd.read_csv(path_valid)
    df = pd.concat([df_train, df_valid], axis=0)

    def make_modality(row):
        '''This is helper function to clean chexpert. It generates a full
        modality string out of other preexisting columns'''
        modality_string = 'XR - Plain Film '
        print('Arranging modality')
        if(isinstance(row['AP/PA'], str)):
            modality_string = modality_string+row['AP/PA']
        if(isinstance(row['Frontal/Lateral'], str)):
            modality_string = modality_string+' '+row['Frontal/Lateral']

        return modality_string

    def make_findings(row):
        """This is helper function to clean chexpert. It generates a full
        findings string out of labels of findings

        """
        print('arranging findings')
        initial_clause_options = ["X-ray demonstrates ",
                                  "X-ray shows ",
                                  "The X-ray study is indicative of ",
                                  "Signs of "]
        negative_clause_options = [
            "There is no evidence of ", "With no signs of ", " There are no signs of "]
        connector_clause_options = [
            "and ", "as well as ", "it also shows ", "there is also "]

        initial_clause = random.choice(initial_clause_options)
        negative_clause = random.choice(negative_clause_options)
        connector_clause = random.choice(connector_clause_options)

        positive_conditions = []
        negative_conditions = []
        for label, value in row[5:].items():
            if not (pd.isna(value) or value == -1):  # avoid ambiguous or unmentioned labels
                if value == 0:
                    negative_conditions.append(label)
                elif value == 1:
                    positive_conditions.append(label)
        findings_string = ''
        # Positive clause
        if len(positive_conditions) > 0:
            findings_string = initial_clause
        for condition in positive_conditions:
            if condition == positive_conditions[-1]:
                findings_string += condition+'.'
            elif (len(positive_conditions) > 1) and (condition == positive_conditions[-2]):
                findings_string += condition + ' ' + connector_clause
            else:
                findings_string += condition + ', '

        # Negative clause
        if len(negative_conditions) > 0:
            findings_string += negative_clause
        for condition in negative_conditions:
            if condition == negative_conditions[-1]:
                findings_string += condition+'. '
            elif (len(negative_conditions) > 1) and (condition == negative_conditions[-2]):
                findings_string += condition + ' or '
            else:
                findings_string += condition + random.choice([', ', ' nor '])
        if len(positive_conditions) == 0 and len(negative_conditions) == 0:
            return 'no findings'
        # print(findings_string)
        return findings_string

    def make_diagnosis(row):
        """This is helper function to clean chexpert.

        It goes through the conditions in the original chexpert data row and
        joins all of them to be presented in natural language format

        Args:
            row (df row): A data frame row from the chexpert raw data

        Returns:
            diagnosis_string (Str)

        """
        print('arranging diagnosis')
        positive_conditions = []
        diagnosis_string = ''
        for label, value in row[5:].items():
            # avoid ambiguous or unmentioned labels
            if not (pd.isna(value) or value == -1):
                if value == 1:
                    positive_conditions.append(label)
        diagnosis_string_list = []
        # Positive clause
        for condition in positive_conditions:
            if condition not in ["No Findings", "No Finding",
                                 "Support Devices", "Lung Opacity"]:
                diagnosis_string_list.append(condition)
        if len(diagnosis_string_list) > 0:
            diagnosis_string = ', '.join(diagnosis_string_list)
        else:
            diagnosis_string = 'no findings'
        # print(diagnosis_string)
        return diagnosis_string
    # We use both helper functions defined above to generate the Modality,
    # Findings and Diagnosis column.

    df["Modality"] = df.apply(make_modality, axis=1)
    df["Findings"] = df.apply(make_findings, axis=1)
    df['Impression'] = df.apply(make_diagnosis, axis=1)

    # Since all anatomy is the same we assign 'chest pulmonary' to Anatomy
    # column
    df['Anatomy'] = 'chest, pulmonary'

    # Relabel so as to have a binary label vector
    # We may need to make a vector that includes 0,-1 and 1 as three
    # different classes but for now we will view them as the presence or
    # absence of a condition.

    # Label processing:
    # columns that should be labels
    label_cols = ['Sex', 'Frontal/Lateral', '']

    # For comparison with other papers, we define strict_label columns as
    # the columns that are usually analyzed in literature when talking about
    # chexpert. The original dataset contains labels:
    #    1:positive,
    #    0:negative
    #    -1: uncertain
    #    nan:not mentioned
    # There are three ways of using the labels described in the paper
    # https://arxiv.org/pdf/1901.07031v1.pdf we will use the multilabel
    # approach for now, where  positive, negative, and uncertain are their
    # own clsses.
    # We will also consider non mentions as negative since it sensible to think
    # if not inlcuded in the report, the conditions are negative.
    # One could make a mask of nan values during training in the future but
    # this will do for now.
    strict_label_cols = ['No Finding', 'Enlarged Cardiomediastinum',
                         'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                         'Edema', 'Consolidation', 'Pneumonia',
                         'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
                         'Pleural Other', 'Fracture', 'Support Devices']
    # Replace nans: non-mentions for 0: negative
    df[strict_label_cols] = df[strict_label_cols].fillna(0)
    df[strict_label_cols] = df[strict_label_cols].replace(0, 'negative')
    df[strict_label_cols] = df[strict_label_cols].replace(1, 'positive')
    df[strict_label_cols] = df[strict_label_cols].replace(-1, 'uncertain')

    # Prepare column names for the multilabel columns
    one_hot = pd.get_dummies(df[strict_label_cols],
                             prefix=['label_' + col for col in strict_label_cols])
    new_cols = list(one_hot.columns)
    # Add new one hot encoded columns
    df[new_cols] = one_hot
    # Eliminate original columns
    df.drop(columns=strict_label_cols, inplace=True)
    # Make paths realtive to src
    prefix = '../data/raw/chexpert/'
    df.Path = prefix+df.Path
    os.makedirs('../data/intermediate', exist_ok=True)
    df.to_csv(save_path, index=False)


def medpix():
    """
    Load medclip dataset and generates a clean version of it.

    Original notebook:
        https://colab.research.google.com/drive/120gockcbGUvgcg43D3XJ2HRY10hkuer0?usp=sharing




    Returns:
        None.

    """
    # Configure where to save the resulting dataset
    save_path_dir = '../data/intermediate/'
    os.makedirs(save_path_dir, exist_ok=True)
    save_path = save_path_dir+'inter_medpix.csv'

    # Read the raw df
    path = '../data/raw/medpix/Dataset_MedPix_V1.xlsx'
    df = pd.read_excel(path)
    # (1) and (2) Eliminate rows where Image_Title has single numbers and
    # eliminate rows where title has word figure , considering that it has
    # either the word 'Figure',  'Film' or 'Infection'
    filter1_2 = (
        (df['Image_Title'].str.contains('Figure', na=False)) |
        (df['Image_Title'].str.contains('Film', na=False)) |
        (df['Image_Title'].str.contains('Infection', na=False)) |
        (df['Image_Title'].str.contains('Replace', na=False))
    )

    df.drop(index=df[filter1_2].index, inplace=True)

    # (3)Eliminate rows where the title has the string "Dermatology Seminar"
    filter3 = (
        (df['Image_Title'].str.contains('Dermatology')) &
        (df['Image_Title'].str.contains('Seminar'))
    )

    df.drop(index=df[filter3].index, inplace=True)

    # (4)Eliminate where diagnosis says New case builder
    filter4 = (
        (df['Diagnosis'].str.contains('New', na=False)) &
        (df['Diagnosis'].str.contains('case')) &
        (df['Diagnosis'].str.contains('builder'))
    )
    df.drop(index=df[filter4].index, inplace=True)

    # (5) Eliminate where diagnosis says Unknown
    filter5 = (df['Diagnosis'].str.contains('Unknown', na=False))
    df.drop(index=df[filter5].index, inplace=True)

    #
    # (7) Eliminate images that have no plane or modality
    df.dropna(axis='index', how='any',
              subset=['Plane', 'Core_Modality'], inplace=True)

    # (8) Eliminate images where no full modality is not provided.
    df.dropna(axis='index', how='any', subset=['Full_Modality'], inplace=True)

    # (9) Do something about the rows containing "Replace with", perhaphs
    # eliminate them.
    filter9 = (
        (df['Caption'].str.contains('Replace', na=False)) &
        (df['Caption'].str.contains('with'))
    )
    df.drop(index=df[filter9].index, inplace=True)

    # (10) --> Check full modality and eliminate cases that contain the words
    # 'Drawing', 'Not specified ', 'Not assigned' and 'Empty'
    filterextra = (
        (df['Full_Modality'].str.contains('Drawing', na=False)) |
        (df['Full_Modality'].str.contains('Not', na=False))
    )
    df.drop(index=df[filterextra].index, inplace=True)

    # Rename columns to fit standard
    df.rename(columns={'Case_Diagnosis': 'Impression', 'Location': 'Anatomy',
                       'Caption': 'Caption', 'ID': 'Path',
                       'Case_URL': 'File URL', 'Image_URL': 'URL',
                       'Full_Modality': 'Modality',
                       'History': 'Patient history'}, inplace=True)

    # Take only relevant columns
    df = df[['Path', 'Modality', 'Anatomy', 'Patient history',
             'Findings', 'Impression', 'Diagnosis']]

    # Make paths relative to source
    prefix = '../data/raw/medpix/Images/'
    suffix = '.jpg'
    df.Path = prefix+df.Path.astype(str)+suffix

    
    
    # Check that the images exist or otherwise eliminate them 
    bad_images=check_images(df.Path.to_list())
    print('listing bad images')
    print(bad_images)
    print(f'there are a total of {len(bad_images)} bad images')
    df=df.loc[~df.Path.isin(bad_images)]
    df.to_csv(save_path)
    return




def check_images(path_list):
    """
    Check integrity of images in path.

    Args:
        path (list of str): list containing the paths to images.

    Returns:
        bad_images (list of str): list of paths pointing to corrupted images.

    """
    process_duration = []  # list to store time for each loop
    iters = 0

    ext_list = ['jpg', 'png', 'jpeg', 'gif', 'bmp']
    bad_images = []
    for f_path in path_list:
        tick = time.time()
        try:
            tip = imghdr.what(f_path)
        except:
            print(f_path+' not found')
            bad_images.append(f_path)
            continue
        if ext_list.count(tip) == 0:

            bad_images.append(f_path)

        if os.path.isfile(f_path):
            try:
                img = cv2.imread(f_path)
                shape = img.shape
            except:
                print('file ', f_path, ' is not a valid image file')
                bad_images.append(f_path)
        else:
            print('could not find file {path}')
        tock = time.time()
        process_duration.append(tock-tick)

        iters = iters+1
        seconds = (len(path_list)-iters)*np.mean(process_duration)
        remaining_time = ','.split(str(datetime.timedelta(seconds=seconds)))[0]
        print(f'Remaining time: {remaining_time}\n' +
              f'Average time per image: {np.mean(process_duration)}')
    return bad_images


def mimic():
    """
    Preprocess mimic dataset
    
    The mimic dataset is split in two directories comming from different sources. 
    These are the mimic-cxr database and mimic-cxr-jpg. This function merges the 
    information in csv files from the orginal datasets to create a new dataframe
    that contains the paths to the images and the medical reports. 
    
    The function also implements two forms of working with different studies. Since
    the original dataset contained studies that had more than 1 image per report, we
    decided to use them in 2 different ways:
        
        1. Divide the two images into different samples, each with their report. These
            instances are labeled in the resulteing dataframe with 'broken' in the 'type' 
            column.
        
        2. Concatenate the two images and represent them as one sample with one report. 
            These instances are labeled in the resulting dataframe with 'multi' in the
            type column
    
    Args:
        None
    Returns
        df (pandas df): A pandas dataframe with paths to images and their corresponding 
            labels and captions. 
    
    """
    def get_concat_h(im1, im2):
        ''' Helper function to prepare mimic.'''
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    
    def merge_studies(df):
        """ 
        Helper function to prepare mimic. 
        Helper function to iterate over the mimic dataset and either brake
        or concatenate images that correspond to the same study. This function 
        is to be called as follows: "df.groupby('study_id').map(merge_studies)"
        
        
        
        """
        global counter
        global times

        tick=time.time()
        # Get the names of the images and the study they belong to. 
        img_names=df['dicom_id'].to_list()
        study_path=df['path'].to_list()[0] # The path here is still not relative to the 
                                           # data folder, it will be modified in the load 
                                           # report section (comments)
        study_id=str(df.study_id.to_list()[0])

        if len(df)>1:
        # If there is more than one image in this study:
            # print(study_path)
            full_img_paths=[]

            #Construct the paths of the images that belong to the same study. 
            for img_name in img_names:
            # For images with that belong to the same study
                full_img_path='../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/'+study_path[:-4]+'/'+img_name+'.jpg'
                full_img_paths.append(full_img_path)
            # print(full_img_paths)


            # Concatenate the images of the same study together. 
            while len(full_img_paths)>1:
                img1=Image.open(full_img_paths.pop(0))
                img2=Image.open(full_img_paths.pop(0))
                concat_path='../data/raw/mimic_fusions/'+study_id+'.jpg'
                concat=get_concat_h(img1,img2).save(concat_path)
                full_img_paths.append(concat_path)

        # Find the report and labels: Since all of the images in the input df belong to one 
        # study, they must have the same report and labels

        # Load report into the contesnts variable
        full_study_path='../data/raw/physionet.org/files/mimic-cxr/2.0.0/'+study_path
        try:
            with open(full_study_path) as f:
                contents = f.readlines()
        except:
            print('problems reading\n'+full_study_path)
            contents=[]
        contents=''.join(contents).strip()
        report = contents

        # Load the labels into a list
        label_cols=['Atelectasis', 'Cardiomegaly',
                   'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
                   'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
                   'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

        labels=df[label_cols].iloc[0].to_list()


        # Make a df with all of the samples for this study:
        # Samples will have 3 categories: 
        #     multi: images in this category are the result of the concatenation of
        #         images of the same study and have only one asociated label and report. 
        #     unique: images in this category are part of a study that only had one 
        #         image as part of it and have only one asociated label and report. 
        #     broken: images in this category belong to a study that had more than one
        #         image in it but they were broken down into individual examples with
        #         repeated information (labels and report) for each of them.  

        rows=[]
        # Make the multi type row
        if len(df)>1:
            multi_row=[concat_path]+labels+[report]+['multi'] # 1 row for merged images
            rows.append(multi_row)

        # Make the broken type rows
            paths=[]
            for img_name in img_names:
            # For images with that belong to the same study
                path='../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/'+study_path[:-4]+'/'+img_name+'.jpg'
                broken_row=[path]+labels+[report]+['broken']
                rows.append(broken_row)
        # Make the unique type row
        if len(df)==1:

            #print(study_path)
            img_path='../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/'+study_path[:-4]+'/'+img_names[0]+'.jpg'
            unique_row = [img_path]+labels+[report]+['unique']
            rows.append(unique_row)
            # print(unique_row)

        # Make a final dataframe with all types of rows.
        rows_df=pd.DataFrame(rows,columns=['path']+label_cols+['report','study_type'])
        # rows_df['study_id']=study_id


        # Calculate and show remaining time to finish computing
        counter = counter-1
        tock=time.time()
        times.append(tock-tick)
        print('Remaining time: {}'.format(str(datetime.timedelta(seconds=counter*np.mean(times)))))
    
        return rows_df
    




    reports_path='../data/raw/physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv'
    labels_path='../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv'
    images_path= '../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv'

    # load dfs
    reports=pd.read_csv(reports_path)
    labels=pd.read_csv(labels_path)
    images=pd.read_csv(images_path)
    
    # Merge datasets to have all images corresponding to each study.
    merged_inner = pd.merge(left=reports,how='inner', right=labels, on='study_id',validate='one_to_one' )
    merged_o_m= pd.merge(left=merged_inner,how='inner',right=images,on='study_id',validate='one_to_many')
    

    os.makedirs('../data/raw/mimic_fusions',exist_ok=True)
    global counter
    global times
    counter=len(merged_o_m.groupby(['study_id']).count())
    times=[]
    merged_o_m.groupby(['study_id']).apply(merge_studies).reset_index(drop=True).to_csv('..data/intermediate/inter_mimic.csv')
    