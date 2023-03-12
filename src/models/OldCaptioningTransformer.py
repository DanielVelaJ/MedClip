#OS
import os

# Tensorflow imports
import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras.layers import MultiHeadAttention
from tensorflow import keras

# Other tensorflow compliments
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
from tensorboard.plugins import projector

# Other python libraries
import numpy as np
import math
import random

import time
import datetime

import re
import textwrap
import json

from matplotlib import pyplot as plt
from tqdm import tqdm as tqdm

from PIL import Image
import csv

# Pandas
import pandas as pd


# Local libraries
from pipelines import build_pipeline, build_coco_pipeline

def make_image_encoder(IMAGE_SIZE=(299,299), backbone='efficientnetb0',trainable=False):
    
    inputs=tf.keras.Input(shape=(*IMAGE_SIZE,3))

    if backbone=='efficientnetb0':
    # Notice that the images must be in float formate (299,299,3) and go from 0 to 255
        print('input images must be floats from 0 to 255 and have (299,299,3) format.') 
        base_model = tf.keras.applications.efficientnet.EfficientNetB0(input_shape=(*IMAGE_SIZE, 3), 
                                                                       include_top=False, 
                                                                       weights="imagenet",)
    if backbone=='resnet50':
    # Special preprocessing needed
     
        base_model=tf.keras.applications.resnet50.ResNet50(input_shape=(*IMAGE_SIZE, 3), 
                                                           include_top=False, 
                                                           weights="imagenet",)
    
    base_model.trainable = trainable
    base_model_out = base_model.output
    base_model_out = tf.keras.layers.Reshape((-1, base_model_out.shape[-1]))(base_model_out)
    cnn_model = tf.keras.models.Model(base_model.input, base_model_out, name='image_encoder')
    return cnn_model


class TransformerEncoder(tf.keras.layers.Layer):
    
    def __init__(self, seq_len, embedding_dim, num_heads,dropout,**kwargs):
        super().__init__(**kwargs)
            

        self.mha = MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim,
                                      dropout=dropout)
        # Mha Q:(T,k_dim), K: (S,key_dim), V (S, v_dim)
        # T=Sequence length of query, S=Number of values to attend to. 
        # notice that there  is one key per value to attend to, meaning 
        # there are S values and S keys. 
        # The operation QxT'xV is performed, which yields dimensions
        # (T x k_dim) x (k_dim x v_dim) = T x v_dim
        # however then the result is projected 
        # back to k_dim size to yield (T x k_dim) result.  
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(embedding_dim, activation='relu'),  # (batch_size, seq_len, dff)
                                        tf.keras.layers.Dense(embedding_dim)])                    # (batch_size, seq_len, d_model)
      
        self.norm_input = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=True, mask=None):

        x=self.norm_input(x)
        attn_output, att = self.mha(query=x, 
                                    value=x, 
                                    key=x, 
                                    return_attention_scores=True)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        
        output_dict={'embeddings':out2,'attentions':att}
        return output_dict

class TextEncoder(object):
    """A text encoder and associated objects for modular implementations.

    The class is to be used within larger architectures such as the clip
    pretraining module or the captioning transformer. The class includes
    the tokenizer as well as encoding functions. 
    
    Usage: 
        encoder=TextEncoder()
        tokens=encoder.tokenizer(df)
        embeddings=encoder.encode(tokens)

    Attributes:
        
        seq_len (int): The sequence length for the outputs. 
        embedding_dim (int): The embedding dimensions. 
        projection_trainable (bool): Whether the projection head of the 
            model is trainable or not. 
        backbone_trainable (bool): Whether the backbone of the model is 
            trainable or not. 
        dropout (float): Dropout rate. 
        tokenizer(function): A tokenizer object built uisng seq_len attribute.  
        encoder(keras_model): An encoder model built on the previous attributes, 
            (seq_len,embedding_dim,projection_trainable and backbone trainable).
        start_token(str): Start token. 
        end_token(str:): End token.
        
        Start and end tokens are added to the begnning and end of captions. 
        
        
    """

    def __init__(self,seq_len=100,embedding_dim=512,projection_trainable=False,
                 backbone_trainable=False,dropout=0.1):
        """Initializes the text encoder to the default values. Builds the encoders
           and tokenizers. 
           
            Args:
                
                seq_len (int): The sequence length for the outputs. 
                embedding_dim (int): The embedding dimensions. 
                projection_trainable (bool): Whether the projection head of the 
                    model is trainable or not. 
                backbone_trainable (bool): Whether the backbone of the model is 
                    trainable or not. 
                dropout(float,optional): The dropout rate. 
               
        """
        self.seq_len=seq_len
        self.embedding_dim=embedding_dim
        self.projection_trainable=projection_trainable
        self.backbone_trainable=backbone_trainable
        self.dropout=dropout
        
        self.tokenizer=self.build_tokenizer()
        self.encoder=self.build_encoder()
    @ property
    def vocab(self):
        print('vocab property not yet implemented')
        vocab=None
        return vocab


    def build_tokenizer(self):
        """Builds the tokenizer according to the class attributes.
        
        Some subclasses may need access to the training data to build the tokenizer. 
        """
        print('build_tokenizer method called but method not implemented.')
        return None
    
    def build_encoder(self):
        """Build the encoder. 
        
        The encoder must take sequences of tokens and return 
        embeddings of size 'self.emb_dim'
        """
        print('build_encoder method called but method not implemented.')
        return None
    
    def tokenize(self,texts):
        """Apply preprocessing and tokenizes the texts.
        
        Args: 
            texts(tensor): Tensor of shape (batch_size,None) of strings. 
            
        Returns: 
            tokens(tensor):Tensor of shape (batch_size,seq_len) of token keys.
        """
       
        
        return self.tokenizer(texts)
        
    def encode(self,tokens,**kwargs):
        """Encode the tokens into vectors of size 'self.embedding_dim'
        Args: 
            tokens(tensor of ints): A tensor of shape (batch_size,self.seq_len)
                containing the token keys. 
        Returns: 
            embeddings(tensor of floats): A tensor of shape 
                (batch_size, self.seq_len, self.embedding_dim)
        """
        return self.encoder(tokens,**kwargs)
        
    def tokenize_and_encode(self,texts):
        """Apply the tokenizer and then the encoder to a sequence of texts. 
        
        Args:
            texts(iterable): A sequence of strings. 
        Returns: 
            embeddings(tensor): A tensor of shape (batch_size,slf.seq_len,self.embedding_dim)
            that contains the embedding representation of the texts. 
            
        """
        tokens=self.tokenize(texts)
        embeddings=self.encode(tokens)
        
        return embeddings
    
    def mask(self,tokens):
        """Mask 0 tokens
        Args: 
            
        """
        return tf.math.not_equal(tokens, 0)
class MatrixTextEncoder(TextEncoder):
    """Children of TextEncoder. Embedding matrix text encoder. 
        
    A simple text encoder composed by a tokenizer and an embedding 
    matrix. 
        
        
        
    """
    def __init__(self,tokenizer_data,**kwargs):
        """Initializes the text encoder to the default values. Builds the encoders
           and tokenizers. 
           
        Args:
            tokenizer_data(iterable): A list or array of texts on which to adapt the 
                tokenizer. 
            embedding_dim (int,optional): The embedding dimensions. Defaults to 512 
            projection_trainable (bool,optional): Whether the projection head of the 
                model is trainable or not. Defaults to False. 
            backbone_trainable (bool,optional): Whether the backbone of the model is 
                trainable or not. Defaults to False. 

        """
        self.tokenizer_data=tokenizer_data
        self.backbone=None
        self.start_token='<start>'
        self.end_token='<end>'
            
        
        # The super class is called at last because I am overriding the build_tokenizer method
        # which now requires self.tokenizer_data so we need to instantiate it first and then 
        # build the rest of attributes from super. 
        super().__init__(**kwargs)
        
    @property
    def vocabulary_size(self):
        "str: The vocabulary size of the tokenizer including OOV token"
        return self.tokenizer.vocabulary_size()
    @property
    def vocabulary(self):
        return self.tokenizer.get_vocabulary()
    @property
    def vocabulary_dict(self):
        dictionary=dict(zip(range(len(self.vocabulary)),self.vocabulary))
        return dictionary
    
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
        
    def build_tokenizer(self):
        """ Builds and adapts the tokenizer according to the class attributes"""
        tokenizer=tf.keras.layers.TextVectorization(
                        max_tokens=None,
                        standardize=self.standarize,
                        split='whitespace',
                        ngrams=None,
                        output_mode='int',
                        output_sequence_length=self.seq_len, # Impotant to fit the ammount of characters in the string. 
                        pad_to_max_tokens=False,
                        vocabulary=None,
                        idf_weights=None,
                        sparse=False,
                        ragged=False,
                        name='Matrix_Text_Encoder_Vectorizer')
        
        tokenizer.adapt(self.tokenizer_data)
        
        return tokenizer
    
    def build_encoder(self):
        """
        Builds a keras functional model text encoder based on the class
        attributes. 
        """
        input_tokens=tf.keras.Input(shape=[None])
        embedding_matrix= tf.keras.layers.Embedding(
            input_dim=self.tokenizer.vocabulary_size(),
            output_dim=self.embedding_dim,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=None,
        )

        output_embeddings=embedding_matrix(input_tokens)
        model=tf.keras.Model(inputs=input_tokens,
                             outputs=output_embeddings,name='Matrix_Text_Encoder')
        if not self.projection_trainable:
            for layer in model.layers:
                layer.trainable=False
        
        return model
    
        
        
class AddLearntPositional(tf.keras.layers.Layer):
    def __init__(self,sequence_length,embedding_dim,**kwargs):
        """Add learnt positional encoding.
        
        """
        
        super().__init__(**kwargs)
       
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, 
            output_dim=embedding_dim)
        
        self.tok_emb_scale=tf.math.sqrt(tf.cast(embedding_dim,tf.float32))
        self.sequence_length=sequence_length

    def call(self,inputs):
        
        length = tf.shape(inputs)[-2] # Length of the sequence
        positions = tf.range(start=0, limit=length, delta=1) # Indexes of the positions
        embedded_positions = self.position_embeddings(positions)
        output_embedding=embedded_positions
        scaled_inputs=self.tok_emb_scale*inputs
        
        return output_embedding+inputs

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self,num_heads,embedding_dim,dropout,debug=False, **kwargs):
        super().__init__(**kwargs)
        
        self.debug=debug
        
        
        # Multi head attention modules
        self.mha1=MultiHeadAttention(num_heads=num_heads,
                                      key_dim=embedding_dim,
                                      dropout=dropout)
        self.mha2=MultiHeadAttention(num_heads=num_heads,
                                     key_dim=embedding_dim,
                                     dropout=dropout)
        
        # Layer normalizations
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # Feed forward network 
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim,activation='relu'),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(embedding_dim),
            tf.keras.layers.Dropout(dropout)])
        
    def call(self,inputs, image_embedding_sequence, mask=False):
        # Mask the inputs if the option to mask inputs is provided
        debug=self.debug
        if debug:
            print('\n\nTransformer decoder called\n-------------------')
        if debug:
            print('calculating masks ')
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(inputs)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
            print('combined mask v2',combined_mask)
            print('padding mask v2', padding_mask)
        if debug: 
            print(f'combined mask: {combined_mask.shape}')
        # Pass inputs through first attention head using masking
        if debug:
            print(f'going through attention head 1:\n\t inputs:{inputs.shape}')
        x = self.mha1(
            query=inputs,
            value=inputs,
            key=inputs, 
            attention_mask=combined_mask
            )
        # Add and normalize
        if debug:
            print(f'going thorugh layer normalization 1:\n\t inputs:{inputs.shape}\n\t x:{x.shape}')
        x1=self.layer_norm1(x+inputs)
        
        # Pass inputs throught second attention head
        if debug:
            print(f'going thorugh mha 2: \n\t x1:{x1.shape}\n\t image\
            _embedding_sequence:{image_embedding_sequence.shape}')
        x2=self.mha2(
            query=x1,
            key=image_embedding_sequence,
            value=image_embedding_sequence,
            attention_mask=padding_mask
            
            )
        # Add and normalize
        if debug:
            print(f'going thorugh addnorm2:\n\t x2:{x2.shape}')
        x2=self.layer_norm2(x1+x2)
        
        # Pass inputs through feed forward network
        if debug:
            print(f'going thorugh feed forward:\n\t x2:{x2.shape}')
        x3=self.feed_forward(x2)
        
        # Add and normalize
        if debug:
            print(f'layer normalization:\n\t x2:{x2.shape}\n\t x3:{x3.shape}')
        x3=self.layer_norm3(x2+x3)
        return x3
    
    
    def get_causal_attention_mask(self, inputs):
        
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        
        mult = tf.concat([tf.expand_dims(batch_size, -1), 
                          tf.constant([1, 1], 
                          dtype=tf.int32)],
                          axis=0,)
        
        return tf.tile(mask, mult)
    

class CaptioningTransformer(tf.keras.Model):
    """Makes a transformer based captioning model.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes:
        image_size (touple): Height and width of the image taken by the model. 
        
        image_encoder (model): A model that takes images (None, H,W,3) 
            and produces a unstacked convolutional volume (None, positions(HxW) ,emb_length)
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self,image_encoder, text_encoder, 
                 transformer_encoder, transformer_decoder,
                 embedding_dim,num_heads,vocab_size,captions_per_image,
                 debug=False,**kwargs):
        
        super().__init__(**kwargs)
        
        self.embedding_dim=embedding_dim
        self.image_encoder=image_encoder
        self.image_projector=tf.keras.layers.Dense(embedding_dim)
        self.text_encoder=text_encoder
        self.text_encoder_encoder=text_encoder.encoder
        self.text_enocder_tokenizer=text_encoder.tokenizer
        self.text_projector=tf.keras.layers.Dense(embedding_dim)
        self.seq_len=text_encoder.seq_len
        self.num_heads=num_heads
        self.add_positional_img= AddLearntPositional(image_encoder.output.shape[-2],self.embedding_dim)
        self.add_positional_txt= AddLearntPositional(self.seq_len,self.embedding_dim)
        self.transformer_encoder=transformer_encoder
        self.transformer_decoder=transformer_decoder
        self.vocab_size=vocab_size
        self.debug=debug
        self.captions_per_image=captions_per_image
    
        # The output model, is responsible of taking the decoder sequence output and 
        # translating it to probabilities of the words in the vocab. For now, 
        # it will be a sigmoid with the vocab size. However we may explore other ways 
        # in the future such as distance metrics or similarity between output embeddings and 
        # target wordd embeddings. 
        self.out=tf.keras.layers.Dense(self.vocab_size,activation='softmax')
        
        # Trackers: These 
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        
   
    @property
    
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]
    
    def train_step(self, batch_data):
        
        debug=self.debug
        # Unpack data 
        if debug:
            print('\n\nCaptioning transformer called\n----------')
        if debug:
            print('unpacking data')
        image_input=batch_data['image']
        text_inputs=batch_data['text']

        batch_acc=0
        batch_loss=0
        for i in range(self.captions_per_image):
            text_input=text_inputs[:,i]
            with tf.GradientTape() as tape:
            # Forward pass
                # Pass text through text encoder
                if debug:
                    print(f'passing text through text encoder:\n\t text_input:{text_input.shape}')

                # Split the predictor text from the target text.
                all_tokens=self.text_encoder.tokenize(text_input)
                predictor_tokens,y_true=all_tokens[:,:-1],all_tokens[:,1:]
                # Get the mask from the targets. 
                
                mask=self.text_encoder.mask(y_true)
                
                if debug:
                    print(f'The mask has dimensions {mask.shape}')

                # Pass image through all image layers
                # (encoding, projection, positional encoding, transformer encoder)
                image_encoding=self.full_encoder(image_input,debug=debug)
                
                # Decode the embeddings using the queries provided by the 
                # predictor tokens.
                y_pred=self.full_decoder(predictor_tokens,image_encoding,mask,debug=debug)

                if debug:
                    print('Calculating loss')
                    print(f'Predictions shape: {y_pred.shape}\ny_true shape{y_true.shape}')

                loss=self.calculate_loss(y_true,y_pred,mask)
                    
            # Compute gradients
            if debug:
                print('computing gradients')
            gradients = tape.gradient(loss, self.trainable_variables)
            
            # Apply gradients
            if debug:
                print('applying weights')
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
            # update batch loss
            batch_loss+=loss
            
            # coumpute accuracy and update batch_accuracy
            if debug:
                print('Computing accuracy')
            acc=self.calculate_accuracy(y_true,y_pred,mask)
            batch_acc+=acc
            
        # Update loss trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc/self.captions_per_image)

        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result()}
      
    def full_encoder(self, image_input,debug=False):
        """
        Apply image encoding layers to image input. 
        
        Apply the image encoder, transformer encoder and image projection layers
        to the image. This is known as the "encoder" in the "encoder-decoder" 
        architecture. Notice that it is called "full-encoder" here because it is
        made of several layers. These are: 
        1. Pass image through image encoder. 
        2. Project image encodings to the dimension of the transformer model. 
        3. Pass projected image encodings through transformer encoder. 
        """
        # Pass image through image encoder. 
        if debug:
            print(f'pass through image encoder:\n\t image_input:{image_input.shape}')
        image_embeddings_0=self.image_encoder(image_input)
        # Project image embeddings to right dimension
        image_embeddings_1=self.image_projector(image_embeddings_0)
        # Add positional encoding
        if debug:
            print(f'Adding positional encoder to image embeddings:\n\t image_embeddings: {image_embeddings_1.shape}')
        image_embeddings_2=self.add_positional_img(image_embeddings_1)

        # Pass image encoder outputs through transformer encoder
        image_embeddings_3=self.transformer_encoder(image_embeddings_2)['embeddings']
        return image_embeddings_3
        
    def full_decoder(self,predictor_tokens,image_embeddings,mask,debug=False):
        """
        Decode the predictor_tokens and the image_embeddings into the
        predictions for words at each slot in the target. 
        
        Args: 
            predictor_tokens: The left part of the training sequence used to 
                predict the right part of the sequence. It must already be 
                tokenized.
            image_embeddings: The result of the full_encoding process of the 
                image. 
            mask: Masking of the empty tokens of the target sequence.
        Returns:
            probs: The probabilities for every word in the output sequence. 
                Probs have shape (None,seq_len,vocab_size)
        """
        # Use text encoder to get embeddings from tokens. 
        token_embeddings=self.text_encoder.encode(predictor_tokens)
        # Project text encodings to right shape for transformer
        if debug:
            print(f'going thorugh text_projector:\n\t embeddings:{token_embeddings.shape}')
        token_embeddings=self.text_projector(token_embeddings)
        # Add positional encoding.
        if debug: 
            print(f'adding positional embeddings to text \n\t embeddings:{token_embeddings.shape}')
        queries=self.add_positional_txt(token_embeddings)
        
         # Go through transformer decoder. 
        if debug:
            print(f'going through decoder\n\t image_embedding_sequence:{image_embeddings.shape}'+
                  f'\n\t text_embedings:{queries.shape} ')
        prediction_context= self.transformer_decoder(queries, 
                                                     image_embedding_sequence=image_embeddings, 
                                                     mask=mask)
        # Compute loss.
        probs=self.out(prediction_context)
        return probs
    
    def calculate_loss(self, y_true, y_pred, mask):
        """
        Calculates the loss and masks it 
        
        Args:
        y_true: Batch of tokens with dimension (None,seq_len-1)
            (-1 because of the left shifting of the input sequence)
        y_pred: A batch of probabilities
        
        """
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        loss= (tf.reduce_sum(loss) / tf.reduce_sum(mask))
        # if math.isnan(loss):
        #     print('loss is nan')
        #     print(f'mask:\n {tf.reduce_sum(mask)}')
        #     print(f'y_true:\n {y_true}')
        #     print(f'y_pred:\n {y_pred}')
        
        return loss
    
    def calculate_accuracy(self, y_true, y_pred, mask):
        # Argmax get's the index of the max across the vocabulary dimension. 
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        
        # The logical and avoids counting correct 
        # predictions of values that are masked
        accuracy = tf.math.logical_and(mask, accuracy)
        
        # Ture values are casted to 1. 
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
   
    def test_step(self,batch_data):
        
        debug=self.debug
        # Unpack data 
        if debug:
            print('\n\nCaptioning transformer called\n----------')
        if debug:
            print('unpacking data')
        image_input=batch_data['image']
        text_inputs=batch_data['text']

        batch_acc=0
        batch_loss=0
        
        for i in range(self.captions_per_image):
            text_input=text_inputs[:,i]
            # Forward pass
            # Pass text through text encoder
            if debug:
                print(f'passing text through text encoder:\n\t text_input:{text_input.shape}')

            # Split the predictor text from the target text.
            all_tokens=self.text_encoder.tokenize(text_input)
            predictor_tokens,y_true=all_tokens[:,:-1],all_tokens[:,1:]
            # Get the mask from the targets. 
            mask=self.text_encoder.mask(y_true)

            # Pass image through all image layers
            # (encoding, projection, positional encoding, transformer encoder)
            image_encoding=self.full_encoder(image_input,debug=debug)

            # Decode the embeddings using the queries provided by the 
            # predictor tokens.
            y_pred=self.full_decoder(predictor_tokens,image_encoding,mask,debug=debug)

            if debug:
                print('Calculating loss')
                print(f'Predictions shape: {y_pred.shape}\ny_true shape{y_true.shape}')

            loss=self.calculate_loss(y_true,y_pred,mask)
                                 
            # update batch loss
            batch_loss+=loss
            
            # coumpute accuracy and update batch_accuracy
            if debug:
                print('Computing accuracy')
            acc=self.calculate_accuracy(y_true,y_pred,mask)
            batch_acc+=acc
            
        # Update loss trackers
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc/self.captions_per_image)

        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result()}
        

def instantiate_captioner(params):
    """
    Instantiates a captioning model. 
    
    Args: 
        params(dict): A dictionary of parameters for the model 
        data_pipeline(pipeline): A pipeline made by the make_pipeline function
    Returns 
        model(tf model): The captioner model
    """
  
    
    
    tokenizer_data=["this is a test"]           # Get the captions to fit tokenizer

    print('Building text encoder')
    text_encoder=MatrixTextEncoder(seq_len=params['SEQ_LEN'],                # Make text encoder
                                   embedding_dim=params['EMBEDDING_DIM'],
                                   tokenizer_data=tokenizer_data,
                                   projection_trainable=True,
                                   backbone_trainable=True)
    
    print('Building rest of the model')
    image_encoder=make_image_encoder(params['IMG_SIZE'],                      # Make image encoder
                                     backbone=params['IMG_BACKBONE'],
                                     trainable= params['IMG_TRAINABLE'])
    
    transformer_encoder=TransformerEncoder(params['SEQ_LEN'],                 # Make transformer encoder
                                           params['EMBEDDING_DIM'],
                                           params['NUM_HEADS'], 
                                           params['DROPOUT'])
    
    transformer_decoder=TransformerDecoder(params['NUM_HEADS'],             # Make transformer decoder
                                           params['EMBEDDING_DIM'],
                                           params['DROPOUT'])

   
    VOCAB_SIZE=text_encoder.vocabulary_size   
   
    model=CaptioningTransformer(image_encoder,                             # Build the captioner model 
                                text_encoder,             
                                transformer_encoder,
                                transformer_decoder,
                                params['EMBEDDING_DIM'],
                                params['NUM_HEADS'],
                                VOCAB_SIZE,captions_per_image=1)
    return model