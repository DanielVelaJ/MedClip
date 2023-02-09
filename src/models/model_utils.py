# Todo: Finish the load_captioning_model function
from models.CaptioningTransformers import CaptioningTransformer
from pipelines import build_pipeline, build_coco_pipeline
import tensorflow as tf
import re
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import textwrap
import json
import pandas as pd
import os
import pickle
import gc



def load_captioning_model(model_path):
    
    # Obtain model configuration from config file.
    
    config_path=model_path+'/'+model_path.split('/')[-1]+'.config'
    with open(config_path, 'rb') as fp:
        config = pickle.load(fp)


    # Instantiate a model 
    model=CaptioningTransformer(config)

    # Obtain some training data
    if config['img_backbone']=='efficientnetb0':
        img=np.random.randint(0,
                              256,
                              (1,*config['img_size']))
    tokens=np.random.randint(0,
                             config['vocab_size'],
                             (1,config['capts_per_img'],config['seq_len']))
    inputs=tf.data.Dataset.from_tensor_slices({'image':img,'tokens':tokens})

    # Train the model for a single batch
    loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction='none',from_logits=False)
    opt = tf.keras.optimizers.Adam()
    model.compile(loss=loss, optimizer=opt, run_eagerly=True)
    model.fit(inputs.batch(1))

    # Load weights into model
    weights_path=model_path+'/'+model_path.split('/')[-1]
    model.load_weights(weights_path) 
    return model