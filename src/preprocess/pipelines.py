# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:03:11 2022

@author: Daniel Vela Jarquin

This module contains the functions to generate pipelines ready for modeling.
it requires the following to work:
    1. The raw folder has the unzipped datasets. This is done by the script
        download_data
    2. The prepared datasets are ready. These are generated by the script
        prepare_data

"""
# TODO:
#  Extract functions from within the make_pipeline function (get definition outside)
    
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


    

def make_pipeline(inter_dataset_path,
                  image_size=(299, 299),downscale=True,
                  shuffle=True,
                  seed=1,fractions=[0.70,0.15,0.15]):
    """
    Generates model_ready training data.

    Takes a prepared (generated by functions in the prepare.py module) dataset
    and builds a dictionary with keys, CLIP, labeling, captioning. The values
    are the datasets in tensorflow format, necessary to run each of the
    pretraining tasks.




    Args:
        inter_dataset_path (str): The path to an intermediate dataset
        
        image_size(touple): Size used to reshape images.
        
        downscale(bool): Whether to divide image values by 255
        shuffle (boolean): optional, defaults to true. If True shuffles the dataset with
            the seed provided as seed argument. If false, not shuffling is applied and 
            train, validation and test are taken sequentially from the lower index onward. 
            
        seed (int): A seed to apply the shuffling for reproducible results. Defaults to 1.
        fractions (list of 3 floats): The list contains 
            [fraction of data for train, fraction for valid, fraction for test]. If not provided
            defaults to [0.7,0.15,0.15]

    Returns:
        datasets_dict(dictionary): A dictionary containing tensorflow datasets.
            The key CLIP contains a tf dataset consisting of elements:
                ({image,text},)
            The key captioning contains a tf dataset consisting of elements:
                (image,caption)
            They key labeling contains a tf dataset consisting of elements:
                (image,labels)

     """
    print('making input pipeline')
    df = pd.read_csv(inter_dataset_path)

    
    # Shuffle dataset
    if shuffle:
        df=df.sample(frac=1,random_state=seed).reset_index()
    #Split into train,valid and split
    n=len(df)
    train_n = int(fractions[0]*n) # Number of samples in training set
    val_n = int(fractions[1]*n) # Number of samples in val set
    test_n = int(fractions[2]*n) # Number of samples in test set
    
    # We need to make sure that there are no duplicates between validation and test sets. 
    # (Except for chexpert dataset)
    # therefore we do the following:
    if 'chexpert' not in inter_dataset_path:
        # Get the values that are unique
        uniques=df[~df.duplicated('Full_Caption')]
        duplicated=df[df.duplicated('Full_Caption')]

        if len(uniques)>=(val_n+test_n):
        # If there are enough unique values to fill the validation and test sets:
            val_df=uniques[0:val_n]
            test_df=uniques[val_n:val_n+test_n]
            rest=uniques[val_n+test_n:] # Samples not used so far

        else: 
            print('not enough unique values to generate these many validation and test samples')

        train_pre=pd.concat([rest,duplicated],axis=0) # values not used so far concatenated with those that have duplicates. 
        train_df=train_pre.sample(frac=1,random_state=seed).reset_index(drop=True) # Shuffle train agian. 
    else:
        train_df=df[0:train_n]
        val_df=df[train_n:train_n+val_n]
        test_df=df[train_n+val_n:train_n+val_n+test_n]
    
    
    # define some functions to load images and manipulate tf.dataset.datasets
    def decode_and_resize(img_path,scale):
        """Recieves an image path, decodes and rescales it. """
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img*scale
        return img

    def to_dict(image, text):
        """ To be called on dataset.map to get a dictionary"""
        return {'image': image, 'text': text}

    final_datasets=[]
    image_paths_list=[]
    captions_list=[]
    for df in [train_df,val_df,test_df]:
    # Make a loop for each train, val, and test split. 
    
        # Make a dataset of images
        # Set whether to leave pixel values between 0-255 or from 0-1
        scale_val = 1
        if downscale:
            scale_val = float(1/255)
        image_paths = df['Path']
        image_paths_list.append(image_paths.to_list())
        images_dataset=tf.data.Dataset.from_tensor_slices(image_paths).map(lambda x: decode_and_resize(x,scale_val),
                                                                             num_parallel_calls=tf.data.AUTOTUNE)
        # Make a dataset with captions
        if ('Full_Caption' in df.columns):
            # If captions are available
            captions = df['Full_Caption'].astype(str)
            # Add start and end tokens to the captions
            # captions.apply(lambda x: '<start> ')
            
            captions_dataset = tf.data.Dataset.from_tensor_slices(captions)
            captions_list.append(captions.to_list())
        else:
            # If captions are not available
            print("Prepared dataframe doestn't contain key: Full_Caption")
            captions_dataset = None

        # Make a dataset of labels
        label_cols = [col for col in df.columns if 'label' in col]
        if (len(label_cols) != 0):
            # If there are labels in the dataset
            labels_dataset = tf.data.Dataset.from_tensor_slices(df[label_cols])
        else:
            labels_dataset = None

        # Generate the proper structure for each pretraining task
        # Proper structure for CLIP
        if (captions_dataset is not None):
            # If there are captions available
            clip_dataset = tf.data.Dataset.zip((images_dataset, captions_dataset))
            clip_dataset = clip_dataset.map(to_dict)
            clip_dataset = clip_dataset.zip((clip_dataset,))
        else:
            clip_dataset = None

        # Proper structure for labeling
        if (labels_dataset is not None):
            # If there are labels available
            labeling_dataset = tf.data.Dataset.zip(
                (images_dataset, labels_dataset))
        else:
            labeling_dataset = None

        # Proper structure for Captioning
        if (captions_dataset is not None):
            # If there are captions available
            captioning_dataset = tf.data.Dataset.zip(
                # Dimensions bellow are expanded for compatibility with models that train 
                # with more than one caption per image. 
                (images_dataset, captions_dataset.map(lambda x:tf.expand_dims(x,axis=0)))) 
            captioning_dataset = captioning_dataset.map(to_dict)
        else:
            captioning_dataset = None

        # append a dictionary with created split to final_datasets
        dataset= {'CLIP':clip_dataset,
                'captioning': captioning_dataset,
                'labeling': labeling_dataset}
        final_datasets.append(dataset)
    
    # Rearange everything in one last dictionary
    final_dictionary={'CLIP':{
                                'train': final_datasets[0]['CLIP'],
                                'val': final_datasets[1]['CLIP'],
                                'test': final_datasets[2]['CLIP'],
                                'train_img_paths': image_paths_list[0],
                                'val_img_paths': image_paths_list[1],
                                'test_img_paths': image_paths_list[2],
                                'train_captions': captions_list[0],
                                'val_captions': captions_list[1],
                                'test_captions': captions_list[2],
        
                             },
                      'captioning':{
                                    'train': final_datasets[0]['captioning'],
                                    'val': final_datasets[1]['captioning'],
                                    'test': final_datasets[2]['captioning'],
                                    'train_img_paths': image_paths_list[0],
                                    'val_img_paths': image_paths_list[1],
                                    'test_img_paths': image_paths_list[2],
                                    'train_captions': captions_list[0],
                                    'val_captions': captions_list[1],
                                    'test_captions': captions_list[2]
                                  },
                     'labeling': {
                                    'train': final_datasets[0]['labeling'],
                                    'val': final_datasets[1]['labeling'],
                                    'test': final_datasets[2]['labeling']
                                 }
                     }
    return final_dictionary


def coco(downscale=True,image_size=(299,299)):
    """
    Generates model_ready training data from the coco dataset.

    Takes the tensorflow datasets instance of the "coco captions" dataset
    and builds a dictionary with keys, 'CLIP' and 'captioning'. The values
    are the datasets (in tensorflow format), necessary to run each of the
    pretraining tasks.




    Args:
        downscale(bool,optional): Whether to divide image values by 255. 
            Defaults to True.
        image_size(touple,optional): Size used to reshape images. Defaults to 
            (299,299)
    

    Returns:
        datasets_dict(dictionary): A dictionary containing tensorflow datasets.
            The key CLIP contains a tf dataset consisting of elements:
                ({image,text},)
            The key captioning contains a tf dataset consisting of elements:
                (image,caption)
            
            There is also a dataset which includes the image paths for each 
            train, val and test split. 
            

     """
    print('Loading coco...')
    print('remember coco dataset has 5 captions per image so',
          ' the value for the "text" key will be a (5,) string array.')
    data,info = tfds.load(name='coco_captions',with_info=True)
    train_imgs = data['train'].map(lambda x:get_coco_image(x,downscale,image_size), 
                                 num_parallel_calls=tf.data.AUTOTUNE)
    train_capts = data['train'].map(get_coco_capts,num_parallel_calls=tf.data.AUTOTUNE)
    train_img_paths=data['train'].map(get_coco_img_paths,num_parallel_calls=tf.data.AUTOTUNE)

    val_imgs = data['val'].map(lambda x:get_coco_image(x,downscale,image_size), 
                                 num_parallel_calls=tf.data.AUTOTUNE)
    val_capts = data['val'].map(get_coco_capts,num_parallel_calls=tf.data.AUTOTUNE)
    val_img_paths=data['val'].map(get_coco_img_paths,num_parallel_calls=tf.data.AUTOTUNE)

    test_imgs = data['test'].map(lambda x:get_coco_image(x,downscale,image_size), 
                                 num_parallel_calls=tf.data.AUTOTUNE) 
    test_capts=data['test'].map(get_coco_capts,num_parallel_calls=tf.data.AUTOTUNE)
    test_img_paths=data['test'].map(get_coco_img_paths,num_parallel_calls=tf.data.AUTOTUNE)

    train_data_captioning=tf.data.Dataset.zip((train_imgs,train_capts)).\
                map(to_dict,num_parallel_calls=tf.data.AUTOTUNE)
    val_data_captioning=tf.data.Dataset.zip((val_imgs,val_capts)).\
                map(to_dict,num_parallel_calls=tf.data.AUTOTUNE)
    test_data_captioning=tf.data.Dataset.zip((test_imgs,test_capts)).\
                map(to_dict,num_parallel_calls=tf.data.AUTOTUNE)

    train_data_clip=tf.data.Dataset.zip((train_data_captioning,))
    val_data_clip=tf.data.Dataset.zip((val_data_captioning,))
    test_data_clip=tf.data.Dataset.zip((test_data_captioning,))


    dataset_dictionary={'CLIP':{
                                    'train': train_data_clip,
                                    'val': val_data_clip,
                                    'test': test_data_clip,
                                    'train_img_paths': train_img_paths,
                                    'val_img_paths': val_img_paths,
                                    'test_img_paths': test_img_paths,
                                    'train_captions': train_capts,
                                    'val_captions': val_capts,
                                    'test_captions': test_capts

                                 },
                          'captioning':{
                                        'train': train_data_captioning,
                                        'val': val_data_captioning,
                                        'test': test_data_captioning,
                                        'train_img_paths': train_img_paths,
                                        'val_img_paths': val_img_paths,
                                        'test_img_paths': test_img_paths,
                                        'train_captions': train_capts,
                                        'val_captions': val_capts,
                                        'test_captions': test_capts
                                      }
                         }
    return dataset_dictionary

def get_coco_image(dataset,downscale=True,image_size=(299,299)):
    """Map function to get an array of images from the coco dataset."""
    img=dataset['image']
    
    img=tf.image.resize(img,image_size)
    if downscale:
        img=img/255
    return img
def get_coco_capts(dataset):
    """Map function to get an array of captions for every image. 
    
    Coco dataset uses 5 captions per image for most images but some have 6 or 7 
    we will crop them to 5 for ease of implementation of batching in the pipeline. 
    Otherwise tensorflow cannot batch different sizes. """
    
    captions=dataset['captions']['text'][0:5]
    return captions

def get_coco_img_paths(dataset):
    """Map function to get the image paths from original coco dataset."""
    return dataset['image/filename']
def to_dict(image, text):
    """ To be called on dataset.map to get a dictionary"""
    return {'image': image, 'text': text}