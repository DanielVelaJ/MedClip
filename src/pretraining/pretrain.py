# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:07:30 2022

@author: danic
"""
# TODO:
import datetime

# Tensorflow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds

# Other pylab imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# local project imports
from preprocess import pipelines


def labeling(data_pipeline, embedding_size, backbone='resnet50', train_backbone=False,
             log_dir='../model_logs/pretraining/labeling/',
             save_model_path='../models/pretraining/labeling/',
             debug=False):
    
    """ Makes a model and pre-trains it in a  multi_class classification problem (meaning each input 
    can be assigned to more than one class). 
    The input and output dimensions are infered from the dataset object. A classifier model 
    (backbone--->linear projection layer--->sigmoids layer)  is created and trained in the given 
    data. 
        
    
    Arguments: 
        dataset (tf dataset): a tensorflow dataset object that outputs a touple (image,label). label 
            must be one-hot encoded. 
        encoding_size (int): The size of the image encoding vector. This is, the amount of neurons in 
            the linear-proyection layer added between the last layer of the backbone and the sigmoid 
            layer.
        backbone (str): the backbone architecture to use. The following are available: 
            'resnet50', 'Xception', etc.
        save_model_path: the path to save the best version of the pre-trained model. 
        reports_path: the path to save the pre-training report and classification metrics. 
    Returns: 
        image_encoder: The trained image encoder which takes the image as an input and outputs a 
            feature vector of the size specified in the 'encoding_size' argument.       
    """
    
    backbonestr=backbone
    date_str=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir=log_dir+date_str
    save_model_path=save_model_path+ date_str

    # Load data
    train_data=data_pipeline['labeling']['train'].batch(128)
    val_data=data_pipeline['labeling']['val'].batch(128)

    # Make the model
    input_shape = train_data.element_spec[0].shape.as_list()[1:]
    n_classes = int(train_data.element_spec[1].shape.as_list()[1])
    inputs=tf.keras.Input(shape=input_shape, name='input')
    if backbone=='resnet50':
        backbone= tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None,
        )
    if (train_backbone==False):
        for layer in backbone.layers:
            layer.trainable=False
    x = tf.keras.applications.resnet50.preprocess_input(inputs) # preprocess data
    x = backbone(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(embedding_size,activation=None,name='encoding_layer')(x)
    x = tf.keras.layers.Dense(n_classes,activation='sigmoid',name='classification_layer')(x)
    model=tf.keras.Model(inputs=inputs,outputs=x, name='labeling_'+backbonestr)


    # Define callbacks
    tf_callback=tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_model_path,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)




    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer='adam',
                  metrics=['accuracy','AUC','Precision','Recall']
                 )
    print('Labeling pretraining initialized with model: \n')
    print(model.summary())
    if debug==True:
        print('Fitting the model for 3 epochs on a debug fraction of the dataset')
        model.fit(train_data.take(2),epochs=3,
                  callbacks= [tf_callback,checkpoint_callback],
                  validation_data=val_data.take(1))
    elif debug==False:
        print('Fitting the model for 20 epochs')
        model.fit(train_data,epochs=20,
                  callbacks=[tf_callback,
                             checkpoint_callback],
                             validation_data=val_data)
        
    
    
    return

def clip(data_pipeline, embedding_size, 
         image_backbone='resnet50',
         text_backbone='bert',
         train_image_backbone=False,
         train_text_backbone=False,
         log_dir='../model_logs/pretraining/clip/',
         save_model_path='../models/pretraining/clip/',
         debug=False):
    """ This function makes and pretrains a text and image encoder 
        on the clip architecture 
        
        Arguments: 
        data_pipeline(dict): a d
        
        
        
        
        
        """

