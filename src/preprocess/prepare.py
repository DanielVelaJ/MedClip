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

import pandas as pd
import random


def chexpert():
    
    '''
    Generates a clean dataframe from chexpert raw data stored in the raw/data
    directory. 

    '''
    chexpert_raw_path = 'data/raw/chexpert'
    path_train = chexpert_raw_path + '/CheXpert-v1.0-small/train.csv'
    path_valid = chexpert_raw_path + '/CheXpert-v1.0-small/valid.csv'
    save_path= 'C:/Users/danic/MedClip/data/intermediate/inter_chexpert.xlsx'

    df_train = pd.read_csv(path_train)
    df_valid = pd.read_csv(path_valid)
    df = pd.concat([df_train, df_valid], axis=0)

    def make_modality(row):
        '''This is helper function to clean chexpert. It generates a full 
        modality string out of other preexisting columns'''
        modality_string = 'XR - Plain Film '
        if(isinstance(row['AP/PA'], str)):
            modality_string = modality_string+row['AP/PA']
        if(isinstance(row['Frontal/Lateral'], str)):
            modality_string = modality_string+' '+row['Frontal/Lateral']

        return modality_string

    def make_findings(row):
    
        '''This is helper function to clean chexpert. It generates a full 
        findings string out of labels of findings'''
        initial_clause_options=["X-ray demonstrates ", "X-ray shows ","The X-ray study is indicative of ","Signs of "]
        negative_clause_options=["There is no evidence of ","With no signs of "," There are no signs of "]
        connector_clause_options=["and ","as well as ","it also shows ","there is also "]
    
        initial_clause=random.choice(initial_clause_options)
        negative_clause=random.choice(negative_clause_options)
        connector_clause=random.choice(connector_clause_options)
    
        positive_conditions=[]
        negative_conditions=[]
        for label,value in row[5:].items():
            if  not (pd.isna(value) or value==-1): #avoid ambiguous or unmentioned labels
                if value==0:
                    negative_conditions.append(label)
                elif value==1: 
                    positive_conditions.append(label)
        findings_string=''
        #Positive clause
        if len(positive_conditions)>0:
            findings_string=initial_clause
        for condition in positive_conditions:
            if condition == positive_conditions[-1]:
                findings_string+=condition+'.'
            elif (len(positive_conditions)>1) and (condition == positive_conditions[-2]):
                findings_string += condition + ' '+ connector_clause
            else:
                findings_string+=condition + ', '
    
        #Negative clause
        if len(negative_conditions)>0:
            findings_string+=negative_clause
        for condition in negative_conditions:
            if condition == negative_conditions[-1]:
                findings_string+=condition+'. '
            elif (len(negative_conditions)>1) and (condition == negative_conditions[-2]):
                findings_string += condition + ' or '
            else:
                findings_string+=condition + random.choice([', ',' nor '])
        if len(positive_conditions)==0 and len(negative_conditions)==0:
            return 'no findings'
        print(findings_string)
        return findings_string
    def make_diagnosis(row):
        '''This is helper function to clean chexpert.    
        It through the conditions in the original chexpert data row and joins
        all of them to be presented in natural language format
        
        Args: 
            row (df row): A data frame row from the chexpert raw data
            
        Returns:
            diagnosis_string (Str)
        
        '''
        positive_conditions=[]
        diagnosis_string=''
        for label,value in row[5:].items():
            if  not (pd.isna(value) or value==-1): #avoid ambiguous or unmentioned labels
                if value==1: 
                    positive_conditions.append(label)
        diagnosis_string_list=[]
        #Positive clause
        for condition in positive_conditions:
            if condition not in ["No Findings","No Finding","Support Devices","Lung Opacity"]:
                diagnosis_string_list.append(condition)
        if len(diagnosis_string_list)>0:
            diagnosis_string=', '.join(diagnosis_string_list)
        else:
            diagnosis_string='no findings'
        print(diagnosis_string)
        return diagnosis_string
    # We use both helper functions defined above to generate the Modality,
    # Findings and Diagnosis column.
    
    df["Modality"] = df.apply(make_modality, axis=1)
    df["Findings"]=df.apply(make_findings,axis=1)
    df['Impression']=df.apply(make_diagnosis,axis=1)
    
    # Since all anatomy is the same we assign 'chest pulmonary' to Anatomy
    # column
    df['Anatomy']='chest, pulmonary'
    df.to_excel(save_path)
def medclip():
    return
