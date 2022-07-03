# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:07:30 2022

@author: danic
"""
import tensorflow as tf


def labeling(dataset, encoding_size, backbone, save_model_path, reports_path):
    
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
    # Load the backbone
    if (backbone == 'resnet50'):
        backbone = tf.keras.applications.resnet50.Resnet50()
        
        
    
    
    return
