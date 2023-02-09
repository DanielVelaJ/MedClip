from pipelines import build_pipeline, build_coco_pipeline
import tensorflow as tf
import re
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os 

class CustomTokenizer(object):
    def __init__(self):
        super().__init__()
        self.vocabulary=None
        self.vocab_size=None
    def build_vocabulary(self, tokenizer_data):
        pass
    def tokenize(self, sequence):
        pass
    def untokenize(self, token_sequence):
        pass
    def use_vocabulary(self,vocab):
        pass
    def save_vocabulary(self):
        pass
    
        
class KerasTokenizer(CustomTokenizer):
    def __init__(self,seq_len):
        super().__init__()
        self.seq_len=seq_len
        # Initialize keras tokenizer
        self.tokenizer=tf.keras.layers.TextVectorization(
            max_tokens=None,
            standardize=self.standarize,
            split='whitespace',
            ngrams=None,
            output_mode='int',
            output_sequence_length=self.seq_len,
            pad_to_max_tokens=False,
            vocabulary=None,
            idf_weights=None,
            sparse=False,
            ragged=False)
    
    def standarize(self, input_string):
        # strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        # strip_chars = strip_chars.replace("<", "")
        # strip_chars = strip_chars.replace(">", "")
        # lowercase = tf.strings.lower(input_string)
        # final_string =tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
        lowercase = tf.strings.lower(input_string)
        
        strip_chars = "!\"#$%&()*+;=?@[\]^_`{|}~"
        final_string =tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
        
        final_string =tf.strings.regex_replace(final_string, "[%s]" % re.escape('.'), " . ")
        final_string =tf.strings.regex_replace(final_string, "[%s]" % re.escape(','), " , ")

        # Replace 1cm and 2mm for 1 cm and 2 mm
        final_string= tf.strings.regex_replace(final_string, 
                                               r'([0-9]+)(cm|mm)', 
                                               r'\1 \2', 
                                               replace_global=True, 
                                               name=None
                                               )
        return final_string 
    
    def build_vocabulary(self,tokenizer_data):
        """
        Args: 
            tokenizer_data(list): list of all strings
        """
        self.tokenizer.adapt(tokenizer_data)
        self.vocab_size = self.tokenizer.vocabulary_size()
        self.vocabulary = self.tokenizer.get_vocabulary()
        return
    
    def tokenize(self, input_string):
        return self.tokenizer(input_string)
    
    def untokenize(self, tokens):
        if tf.is_tensor(tokens):
            tokens_list=list(tokens.numpy())
            
        if isinstance(tokens, np.ndarray):
            tokens_list=list(tokens)
        
        
        output_string_list=[]        
        for token in tokens_list: 
            output_string_list.append(self.vocabulary[token])
        output_string=' '.join(output_string_list).strip()
        return output_string
    
        
        
    def save_vocabulary(self, model_path):
        """Save a vocabulary to the vocabularies directory"""
        os.makedirs(model_path, exist_ok=True)
        vocabulary_path=model_path+'/'+model_path.split('/')[-1]+'.vocabulary'
        with open(vocabulary_path, 'wb') as fp:
            pickle.dump(self.vocabulary, fp)
        return
   
        
                             
    def load_vocabulary(self, model_path):
        """Save a vocabulary to the vocabularies directory"""
        vocabulary_path=model_path+'/'+model_path.split('/')[-1]+'.vocabulary'
        with open(vocabulary_path, 'rb') as fp:
            vocabulary = pickle.load(fp)
        self.vocabulary=vocabulary
        self.tokenizer.set_vocabulary(vocabulary, idf_weights=None)
        return 

