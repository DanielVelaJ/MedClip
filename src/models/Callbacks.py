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

class CaptioningCallback(tf.keras.callbacks.Callback):
    """ A callback that logs the paremeter dictionary of the model.
    """
    def __init__(self,config,log_dir):
        """
        Initialize the callback. 
        
        Args:
            config(dict): The parameters of the model.
            log_dir(str): Path to the model tensorboard log.  
        """
        super().__init__()
        self.model_config=config
        self.log_dir=log_dir
        
    def on_train_begin(self,logs=None):
        """
        Log the model parameters. 

        """
        # Log model parameters to tensorboard.
        file_writer = tf.summary.create_file_writer(self.log_dir) 
        with file_writer.as_default():
            tf.summary.text("model_config", self.pretty_json(self.model_config), step=0)
            
        # Save configuration file.
#         os.makedirs(self.model_path,exist_ok=True)
#         config_path=self.model_path+'/'+self.model_path.split('/')[-1]+'.config'
        
#         with open(config_path, 'wb') as fp:
#             pickle.dump(self.model_config, fp)
            
    def on_epoch_end(self, epoch,logs):
        ''' 
        Log bleu metrics (not implemented yet).
        '''
        pass
    
    def pretty_json(self,hp):
        """Dumps dictionary into a formatted string.  
        
        Args: 
           hp(dict): The parameters dictionary
        
        Returns:
            json_hp(str): A formatted string with the contents of the hp dict
        """
        json_hp = json.dumps(hp, indent=2)
        return "".join("\t" + line for line in json_hp.splitlines(True))