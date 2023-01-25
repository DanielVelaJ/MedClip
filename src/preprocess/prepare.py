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
from tqdm import tqdm

def assign_group(row, grouping):
    """Given a dictionary of groupings, return the corresponding group.
    
    This function is to be used within a call to an .apply() method for 
    a pandas series. 
    """
    for group,values in grouping.items():
        if row in values:
            return group

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
        # print('Arranging modality')
        if(isinstance(row['AP/PA'], str)):
            modality_string = modality_string+row['AP/PA']
        if(isinstance(row['Frontal/Lateral'], str)):
            modality_string = modality_string+' '+row['Frontal/Lateral']

        return modality_string

    def make_findings(row):
        """This is helper function to clean chexpert. It generates a full
        findings string out of labels of findings

        """
        # print('arranging findings')
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
    
    # Build the Full Caption column to be predicted by models
    df['Full_Caption']=df.apply(lambda row: ('<start>'+
                                         ' Modality: ' + str(row['Modality'])+
                                         ' Anatomy: ' + str(row['Anatomy'])+
                                         ' Findings: '+ str(row['Findings'])+
                                         # ' Impression: '+ str(row['Impression'])+
                                             ' <end>') ,axis=1)
    
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

    # Grab useful columns
    # useful_cols=['ID','Plane',
    #              'Core_Modality','Full_Modality',
    #              'Findings','Case_Diagnosis','Location']
    useful_cols=['ID','Plane',
                 'Core_Modality','Full_Modality',
                 'Caption','Case_Diagnosis','Location']
    
    df=df[useful_cols]

    # Rename columns
    renamings={'ID':'Path',
               'Full_Modality':'Modality',
              'Case_Diagnosis':'Impression',
              'Location':'Anatomy'}
    df.rename(columns=renamings,inplace=True)
    # Now the dataframe contains columns 
    #['Path', 'Plane', 'Core_Modality', 'Modality', 'Caption', 'Impression',
    # 'Anatomy']


    # Drop rows with empty values
    df = df.dropna() 

    # CLEANING ON THE "PLANE" COLUMN-------------------------
    # Consolidating synonims into single types.
    df.Plane.replace('Transverse','Axial',inplace=True)
    df.Plane.replace('Lateral','Sagittal',inplace=True)
    df.Plane.replace('Frontal','Coronal',inplace=True)

    # Keep rows that have a plane values with a frequency higher than 100
    valid_index=df.Plane.value_counts().index[df.Plane.value_counts()>100]
    df=df.loc[df.Plane.isin(valid_index)]

    # Drop whenever plane is equal to particular values.
    df = df.loc[~df.Plane.isin(['NOS - Not specified', #
                          'Other View (see caption)'])]

    # CLEANING ON THE "CORE_MODALITY" COLUMN-------------------
    # Consolidating synonims under the same concept.
    df.Core_Modality.replace('US-D','US',inplace=True)
    df.Core_Modality.replace('CTA','AN',inplace=True)
    df.Core_Modality.replace('MRA','AN',inplace=True)
    df.Core_Modality.replace('Histology','HE',inplace=True)
    df.Core_Modality.replace('PET','PET/NM',inplace=True)
    df.Core_Modality.replace('NM','PET/NM',inplace=True)
    df.Core_Modality.replace('PET-CT','PET/NM',inplace=True)
    df.Core_Modality.replace('MRS','MR',inplace=True)
    
    # Renaming interventional instances to INT
    df.Core_Modality.replace('Interventional','INT',inplace=True) 

    # Keep rows that have a Core_Modality values with a frequency higher than 100
    valid_index=df.Core_Modality.value_counts().index[df.Core_Modality.value_counts()>100]
    df=df.loc[df.Core_Modality.isin(valid_index)]

    # Drop whenever plane is equal to particular values.
    df = df.loc[~df.Core_Modality.isin(['NOS'])]

    # CLEANING ON "FINDINGS" COLUMN------------------------------
    # Eliminate rows that have a "findings" wordcount larger than 100 words. 
    df["Number of Words"] = df["Caption"].apply(lambda n: len(n.split()))
    df=df.loc[df['Number of Words']<=100]

    # CLEANING ON THE "ANATOMY" COLUMN--------------------------
    # Consolidation
    df.Anatomy.replace('Brain and Neuro','Brain',inplace=True)
    df.Anatomy.replace('Nerve, central','Brain',inplace=True)

    df.Anatomy.replace('MSK - Musculoskeletal','Musculoskeletal',inplace=True)
    df.Anatomy.replace('Extremity - Please Select MSK','Musculoskeletal',inplace=True)

    df.Anatomy.replace('Chest, Pulmonary (ex. Heart)','Pulmonary',inplace=True)

    df.Anatomy.replace('Breast and Mammography','Breast',inplace=True)

    df.Anatomy.replace('Abdomen - Generalized','Abdomen',inplace=True)
    df.Anatomy.replace('Gastrointestinal','Abdomen',inplace=True)

    df.Anatomy.replace('Head and Neck (ex. orbit)','Head and Neck',inplace=True)
    df.Anatomy.replace('Eye and Orbit (exclude Ophthalmology)','Head and Neck',inplace=True)

    df.Anatomy.replace('Vascular','Cardiovascular',inplace=True)
    df.Anatomy.replace('Cardiovascular (inc. Heart)','Cardiovascular',inplace=True)

    df.Anatomy.replace('Multisystem','Generalized',inplace=True)

    # Keep rows that have Anatomy values with a frequency higher than 200
    valid_index=df.Anatomy.value_counts().index[df.Anatomy.value_counts()>200]
    df=df.loc[df.Anatomy.isin(valid_index)]

    # CLEANING THE IMPRESSIONS COLUMN---------------------
    # Keep impressions with at most 30 words.
    df["Number of Words"] = df["Impression"].apply(lambda n: len(n.split()))
    df=df.loc[df['Number of Words']<=30]

    # Eliminate the Number of words column
    df.drop(columns='Number of Words').count()

    # CONVERT THE PATH COLUMN INTO THE COMPLETE PATHS--------------------
    prefix = '../data/raw/medpix/Images/'
    suffix = '.jpg'
    df.Path = prefix+df.Path.astype(str)+suffix

    # CREATE THE FULL CAPTIONS COLUMN-------------------------------------
    df['Full_Caption']=df.apply(lambda row: ('<start>'+
                                             ' Core Modality: '+ str(row['Core_Modality'])+
                                             ' Modality: ' + str(row['Modality'])+
                                             ' Plane: ' + str (row['Plane']) +
                                             ' Anatomy: ' + str(row['Anatomy'])+
                                             ' Findings: '+ str(row['Caption'])+
                                             #' Impression: '+ str(row['Impression'])+
                                             ' <end>') ,axis=1)

    # CHECK THAT WE ARE ABLE TO OPEN IMAGES POINTED BY THE PATH COLUMN------------------
    bad_images=check_images(df.Path.to_list())
    print('listing bad images')
    print(bad_images)
    print(f'there are a total of {len(bad_images)} bad images')
    df=df.loc[~df.Path.isin(bad_images)] # eliminate rows with bad images from dataframe
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


    ext_list = ['jpg', 'png', 'jpeg', 'gif', 'bmp']
    bad_images = []
    for f_path in tqdm(path_list):
        try:
            tip = imghdr.what(f_path)
        except:
            # print(f_path+' not found')
            bad_images.append(f_path)
            continue
        if ext_list.count(tip) == 0:

            bad_images.append(f_path)

        if os.path.isfile(f_path):
            try:
                img = cv2.imread(f_path)
                shape = img.shape
            except:
                # print('file ', f_path, ' is not a valid image file')
                bad_images.append(f_path)
        else:
            pass
            # print('could not find file {path}')

    return bad_images


def mimic():
    """Obtain clean dataframe for mimic dataset.
    """
    def process_mimic(row): 
        """Process a row in the cxr-record-list.csv dataframe to 
        include caption and complete path to jpg image. 
        """
        try:
        #Do all of the following inside try to avoid lengthy stops
            text_path_prefix='../data/raw/physionet.org/files/mimic-cxr/2.0.0/'
            image_path_prefix= '../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/'
            text_path_suffix=texts_df.loc[texts_df['study_id']==row['study_id']]['path'].iloc[0]

            # Obtain the report ---------------------------
            text_path=text_path_prefix+text_path_suffix
            text_file = open(text_path, "r") # Read report into string
            report = text_file.read()  #read whole file to a string
            text_file.close() #close file
            row['Full_Caption']='<start> '+ report+' <end>' # Add report to row

            # get image path------------------------------
            row['Path']=image_path_prefix+row['path'][0:-3]+'jpg' # Add image path to row.

            return row[['dicom_id','Path','Full_Caption']] # Return only selected columns of row. 
        except:
        # If the process fails then writ error into Full_Caption and Path columns. 
            row['Full_Caption']='error'
            row['Path']='error'
            return row[['dicom_id','Path','Full_Caption']]
    
    images_df_path="../data/raw/physionet.org/files/mimic-cxr/2.0.0/cxr-record-list.csv"
    texts_df_path="../data/raw/physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv"
    labels_df_path="../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"


    images_df=pd.read_csv(images_df_path)
    texts_df=pd.read_csv(texts_df_path)
    labels_df=pd.read_csv(labels_df_path)
    tqdm.pandas()
    df=images_df.progress_apply(process_mimic,axis=1)
    df.to_csv('../data/intermediate/mimic.csv')

# def mimic():
#     """
#     Preprocess mimic dataset
    
#     The mimic dataset is split in two directories comming from different sources. 
#     These are the mimic-cxr database and mimic-cxr-jpg. This function merges the 
#     information in csv files from the orginal datasets to create a new dataframe
#     that contains the paths to the images and the medical reports. 
    
#     The function also implements two forms of working with different studies. Since
#     the original dataset contained studies that had more than 1 image per report, we
#     decided to use them in 2 different ways:
        
#         1. Divide the two images into different samples, each with their report. These
#             instances are labeled in the resulteing dataframe with 'broken' in the 'type' 
#             column.
        
#         2. Concatenate the two images and represent them as one sample with one report. 
#             These instances are labeled in the resulting dataframe with 'multi' in the
#             type column
    
#     Args:
#         None
#     Returns
#         df (pandas df): A pandas dataframe with paths to images and their corresponding 
#             labels and captions. 
    
#     """
#     def get_concat_h(im1, im2):
#         ''' Helper function to prepare mimic.'''
#         dst = Image.new('RGB', (im1.width + im2.width, im1.height))
#         dst.paste(im1, (0, 0))
#         dst.paste(im2, (im1.width, 0))
#         return dst
    
#     def merge_studies(df):
#         """ 
#         Helper function to prepare mimic. 
#         Helper function to iterate over the mimic dataset and either brake
#         or concatenate images that correspond to the same study. This function 
#         is to be called as follows: "df.groupby('study_id').map(merge_studies)"
        
        
        
#         """
#         global counter
#         global times

#         tick=time.time()
#         # Get the names of the images and the study they belong to. 
#         img_names=df['dicom_id'].to_list()
#         study_path=df['path'].to_list()[0] # The path here is still not relative to the 
#                                            # data folder, it will be modified in the load 
#                                            # report section (comments)
#         study_id=str(df.study_id.to_list()[0])

#         if len(df)>1:
#         # If there is more than one image in this study:
#             # print(study_path)
#             full_img_paths=[]

#             #Construct the paths of the images that belong to the same study. 
#             for img_name in img_names:
#             # For images with that belong to the same study
#                 full_img_path='../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/'+study_path[:-4]+'/'+img_name+'.jpg'
#                 full_img_paths.append(full_img_path)
#             # print(full_img_paths)


#             # Concatenate the images of the same study together. 
#             while len(full_img_paths)>1:
#                 img1=Image.open(full_img_paths.pop(0))
#                 img2=Image.open(full_img_paths.pop(0))
#                 concat_path='../data/raw/mimic_fusions/'+study_id+'.jpg'
#                 concat=get_concat_h(img1,img2).save(concat_path)
#                 full_img_paths.append(concat_path)

#         # Find the report and labels: Since all of the images in the input df belong to one 
#         # study, they must have the same report and labels

#         # Load report into the contesnts variable
#         full_study_path='../data/raw/physionet.org/files/mimic-cxr/2.0.0/'+study_path
#         try:
#             with open(full_study_path) as f:
#                 contents = f.readlines()
#         except:
#             print('problems reading\n'+full_study_path)
#             contents=[]
#         contents=''.join(contents).strip()
#         report = contents

#         # Load the labels into a list
#         label_cols=['Atelectasis', 'Cardiomegaly',
#                    'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture',
#                    'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
#                    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

#         labels=df[label_cols].iloc[0].to_list()


#         # Make a df with all of the samples for this study:
#         # Samples will have 3 categories: 
#         #     multi: images in this category are the result of the concatenation of
#         #         images of the same study and have only one asociated label and report. 
#         #     unique: images in this category are part of a study that only had one 
#         #         image as part of it and have only one asociated label and report. 
#         #     broken: images in this category belong to a study that had more than one
#         #         image in it but they were broken down into individual examples with
#         #         repeated information (labels and report) for each of them.  

#         rows=[]
#         # Make the multi type row
#         if len(df)>1:
#             multi_row=[concat_path]+labels+[report]+['multi'] # 1 row for merged images
#             rows.append(multi_row)

#         # Make the broken type rows
#             paths=[]
#             for img_name in img_names:
#             # For images with that belong to the same study
#                 path='../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/'+study_path[:-4]+'/'+img_name+'.jpg'
#                 broken_row=[path]+labels+[report]+['broken']
#                 rows.append(broken_row)
#         # Make the unique type row
#         if len(df)==1:

#             #print(study_path)
#             img_path='../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/'+study_path[:-4]+'/'+img_names[0]+'.jpg'
#             unique_row = [img_path]+labels+[report]+['unique']
#             rows.append(unique_row)
#             # print(unique_row)

#         # Make a final dataframe with all types of rows.
#         rows_df=pd.DataFrame(rows,columns=['path']+label_cols+['report','study_type'])
#         # rows_df['study_id']=study_id


#         # Calculate and show remaining time to finish computing
#         counter = counter-1
#         tock=time.time()
#         times.append(tock-tick)
#         print('Remaining time: {}'.format(str(datetime.timedelta(seconds=counter*np.mean(times)))))
    
#         return rows_df
    




#     reports_path='../data/raw/physionet.org/files/mimic-cxr/2.0.0/cxr-study-list.csv'
#     labels_path='../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv'
#     images_path= '../data/raw/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv'

#     # load dfs
#     reports=pd.read_csv(reports_path)
#     labels=pd.read_csv(labels_path)
#     images=pd.read_csv(images_path)
    
#     # Merge datasets to have all images corresponding to each study.
#     merged_inner = pd.merge(left=reports,how='inner', right=labels, on='study_id',validate='one_to_one' )
#     merged_o_m= pd.merge(left=merged_inner,how='inner',right=images,on='study_id',validate='one_to_many')
    

#     os.makedirs('../data/raw/mimic_fusions',exist_ok=True)
#     global counter
#     global times
#     counter=len(merged_o_m.groupby(['study_id']).count())
#     times=[]
#     merged_o_m.groupby(['study_id']).apply(merge_studies).reset_index(drop=True).to_csv('..data/intermediate/inter_mimic.csv')
    