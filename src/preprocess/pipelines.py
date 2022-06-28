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
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def make_pipeline(inter_dataset_path,
                  image_size=(299, 299)):
    """
    Generates model_ready training data.

    Builds a dictionary with keys, CLIP, labeling, captioning. The values
    are the datasets in tensorflow format, necessary to run each of the
    pretraining tasks.




    Args:
        inter_dataset_path (str): The path to an intermediate dataset
        image_shape(touple): Size used to reshape images.

    Returns:
        datasets_dict(dictionary): A dictionary containing tensorflow datasets.
        The key CLIP contains a tf dataset consisting of elements:
            ({image,text},)
        The key captioning contains a tf dataset consisting of elements:
            (image,caption)
        They key labeling contains a tf dataset consisting of elements:
            (image,labels)

     """
# Parameters

    inter_dataset_path = 'C:/Users/danic/MedClip/data/intermediate/inter_chexpert.csv'
    IMAGE_SIZE = (299, 299)

    df = pd.read_csv(inter_dataset_path)

    def decode_and_resize(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img/255
        return img

    def to_dict(image, text):
        return {'image': image, 'text': text}

    # Make a dataset of images
    image_paths = df['Path']
    images_dataset = tf.data.Dataset.from_tensor_slices(image_paths).map(decode_and_resize,
                                                                         num_parallel_calls=tf.data.AUTOTUNE)

    # Make a dataset with captions
    captions = df['Findings']
    captions_dataset = tf.data.Dataset.from_tensor_slices(captions)

    # Make a dataset of labels
    label_cols = [col for col in df.columns if 'label' in col]
    labels_dataset = tf.data.Dataset.from_tensor_slices(df[label_cols])

    # Generate the proper structure for each pretraining task

    # Proper structure for CLIP
    clip_dataset = tf.data.Dataset.zip((images_dataset, captions_dataset))
    clip_dataset = clip_dataset.map(to_dict)
    clip_dataset = clip_dataset.zip((clip_dataset, labels_dataset))

    # Proper structure for labeling
    labeling_dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))

    # Proper structure for Captioning
    captioning_dataset = tf.data.Dataset.zip(
        (images_dataset, captions_dataset))

    return {'CLIP': clip_dataset,
            'captioning': captioning_dataset,
            'labeling': labeling_dataset}
