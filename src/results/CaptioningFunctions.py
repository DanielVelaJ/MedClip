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

def predict_caption(img,model,tokenizer,show=False):
    """ Uses the given image, model and tokenizer to 
    produce a caption. 
    Args: 
        img: (299,299,3) array of floats. 
        model: a CaptioningTransformer.
        tokenizer: A CustomTokenizer son.
        show(optional bool): Whether to print results. 
        
    Returns:
    
        
    """
    img=tf.expand_dims(img,0) # Add batch dimension to image. 
    # Initializing output to <start> token
    pred_tokens=tokenizer.tokenize('<start>')[0:-1]
    # Adding batch dimension and converting to numpy
    pred_tokens=tf.expand_dims(pred_tokens,0).numpy()
    for i in range(model.config['seq_len']-2):
    # For every prediction step in the sequence:
        mask=tf.math.not_equal(pred_tokens,0) # get mask
        probs=model.calculate_probs(img,pred_tokens,mask) # obtain next word probs
        # obtain the vocab index/token of the next highest probable word for the 
        # current prediction step [i]
        prediction=tf.argmax(probs,-1)[0][i].numpy() 
        if prediction==tokenizer.tokenize('<end>').numpy()[0]:
        # If the prediction matches the end token stop the process. 
            break
        pred_tokens[0,i+1]=prediction # Append the predicted token to the output
    # convert predicted token sequence to words. 
    pred_caption=tokenizer.untokenize(pred_tokens[0][1:])
    if show==True:
        plt.imshow(img.numpy()[0].astype(int))
        plt.show()
        print(pred_caption)
        
    return pred_caption

def caption_dataset(dataset,model,tokenizer,model_name='test_model',show=False):
    """ Make captions for all the images in the dataset. 
    """
    d={'predicted_caption':[]}
    for i,s in enumerate(dataset):
        img=s['image'] # Obtain image
        # predict caption and save it to dictionary.
        d['predicted_caption'].append(predict_caption(img,model,
                                          tokenizer,
                                          show=True))
        # Get the true token sequences there may be more 
        # than one per image. 
        true_tokens=s['tokens']
        for j in range(true_tokens.shape[0]):
        # For every caption in the group of captions for the same image.
            # Untokenize the caption while excluding the start token
            caption=tokenizer.untokenize(true_tokens[j][1:])
            # Eliminate the <end> word in the untokenized caption.
            # and strip white spaces. 
            caption=caption[0:caption.index('<end>')].strip()
            # Save caption to dictionary. 
            col_name='true_caption_'+str(j+1)
            if i==0:
            # If this is the first time the key is used create list.
                d[col_name]=[caption]
            else: 
            # Otherwise append to list. 
                d[col_name].append(caption)
                
        results=pd.DataFrame(d)
        os.makedirs(f'../results/{model_name}/',exist_ok=True)
        results.to_csv(f'../results/{model_name}/results_'+model_name+'.csv',index=False)