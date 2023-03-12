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

# The encoder layer

# The image encoder class
class ImageEncoder(tf.keras.layers.Layer):
    def __init__(self,config):
        """Image encoder class. 
        
        The layer takes images and returns a sequence of vectors 
        corresponding to the last convolutional volume of the given 
        backbone network after being projected to the model's dimension
        "model_dim". 
        
        Args: 
            config(dict): A configuration dictionary. Must contain keys:
                'img_backbone_trainable': whether the backbone is trainable
                'img_size': the resolution of the image ex: (299,299,3)
                'img_backbone': Currently only efficientnetB0 is implemented
                
        """
        super().__init__()
        # Base attrobites 
        self.backbone = config['img_backbone']
        self.img_size = config['img_size']
        self.img_backbone_trainable = config['img_backbone_trainable']
        
        # Initialize backbone model
        self.backbone_model = None # initialized bellow in init_backbone()
        self.init_backbone()
        
        # Other model layers
        self.reshape = tf.keras.layers.Reshape(
            (-1,self.backbone_model.output.shape[-1])
        )
        self.projection = tf.keras.layers.Dense(config['model_dim'])
        
    def call(self,inputs):
        x=self.backbone_model(inputs)
        x=self.reshape(x)
        x=self.projection(x)
        return x
    
    def init_backbone(self):
        """Initializes the backbone model."""
        
        if self.backbone=='efficientnetb0':
            # Notice that the images must be in float format (299,299,3) and go from 0 to 255
            self.backbone_model = tf.keras.applications.efficientnet.EfficientNetB0(
                input_shape=self.img_size, include_top=False, weights="imagenet",
            )
        # Implement other backbones here
        else: 
            # If backbone is not recognized.
            print(f'backbone {self.backbone} not implemented')
        
        # Set backbone model trainability. 
        self.backbone_model.trainable = self.img_backbone_trainable
        
        
class PositionalEncoder(tf.keras.layers.Layer):
    """A layer that adds positional encoding to a squence of embeddings."""
    
    def __init__(self,config,seq_len,**kwargs):   
        """Initialize a positional encoder. 
        
        Creates an learnable embedding matrix to represent positions for
        positional encoding. It also scales the resulting embeddings by the 
        square_root of the model dimension. 
        
        Args: 
            config(dict): A configuration dictionary that must contain: 
                model_dim: The size of the embeddings. Typically 728.
            sequence_length: The size of the sequence for positional encoding. 
        
        
        """
        super().__init__(**kwargs)
        
        
        self.sequence_length=seq_len
        # Embedding layer that generates an embedding for 
        # every possible position in the sequence
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=self.sequence_length, output_dim=config['model_dim']
        )
        # Scaling value = sqrt(model_dim)
        self.scale=tf.math.sqrt(tf.cast(config['model_dim'],tf.float32))

    def call(self,inputs):
        scaled_inputs=self.scale*inputs # scale inputs
        # Generate a positions array filled with
        # numbers from 0 to seq_length
        positions = tf.range(start=0, limit=self.sequence_length, delta=1) 
        # Create embeddings for every index in the positions array.
        embedded_positions = self.position_embeddings(positions)
        
        return embedded_positions+scaled_inputs
    
class FeedForward(tf.keras.layers.Layer):
    """The feed forward network of the transformer. 
    
    A dense feed forward network that operates on each embedding individually. 
    It is composed by two linear layers with a gelu activation in between. 
    
    """
    def __init__(self,config,**kwargs):
        """Initializes layers for feedforward network.
        
        Args:
            config (dict): A configuration dictionary.
                dfadsf.
                
        """
        super().__init__(**kwargs)
        self.linear_1 = tf.keras.layers.Dense(config['intermediate_size'],
                                             activation='gelu')
        self.linear_2 = tf.keras.layers.Dense(config['model_dim'])
        
        self.dropout = tf.keras.layers.Dropout(config['dropout'])
    
    def call(self,inputs):
        x=self.linear_1(inputs)
        x=self.linear_2(x)
        x=self.dropout(x)
        return x
        
class TransformerEncoder(tf.keras.layers.Layer):
    """The transformer encoder part without positional encoding. """
    def __init__(self,config,**kwargs):
        super().__init__( **kwargs)
        self.layer_norm_1=tf.keras.layers.LayerNormalization()
        self.layer_norm_2=tf.keras.layers.LayerNormalization()
        self.mha=tf.keras.layers.MultiHeadAttention(
            num_heads=config['num_heads'],
            key_dim=config['model_dim'], # The dimensions of the dot product for attention in every head
            dropout=config['dropout']
        )
        self.ffw=FeedForward(config)
    def call(self,inputs):
        x=self.mha(query = inputs,
                   value = inputs,
                   key = inputs)
        x1=self.layer_norm_1(x+inputs)
        x=self.ffw(x1)
        x=self.layer_norm_2(x+x1)
        return(x)
    
class FullEncoder(tf.keras.layers.Layer):
    """"The full encoder containing the image encoder, positional encoder and 
    transformer encoder"""
    def __init__(self,config,**kwargs):
        super().__init__(**kwargs)
        self.num_encoder_blocks=config['num_encoder_blocks']

        self.image_encoder = ImageEncoder(config)
        self.positional_encoder = PositionalEncoder(config,seq_len=100)
        self.transformer_encoders = [TransformerEncoder(config) 
                                     for _ in range(config['num_encoder_blocks'])]
    def call(self,inputs):
        x=self.image_encoder(inputs)
        x=self.positional_encoder(x)
        for i in range(self.num_encoder_blocks):
            x=self.transformer_encoders[i](x)
        return x


class TextEmbeddings(tf.keras.layers.Layer):
    
    def __init__(self,config,**kwargs):
        """Receive tokens, embedd them and positionally encode them. 
        
        Args: 
            config(dict): a configuration dictionary
            
        """
        super().__init__(**kwargs)
        
        self.embeddings=tf.keras.layers.Embedding(config['vocab_size'],
                                                   config['model_dim'])
        self.positional=PositionalEncoder(config,seq_len=config['seq_len']-1)
        
    def call(self,inputs):
        # Inputs arrive as token sequences which are 
        # embedded by the "embeddings" matrix. 
        x=self.embeddings(inputs)
        x=self.positional(x) # Add positional encoding.
        return x
        
class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self,config, **kwargs):
        super().__init__(**kwargs)
        self.mha_1=tf.keras.layers.MultiHeadAttention(num_heads=config['num_heads'],
                                                    key_dim=config['model_dim'],
                                                    dropout=config['dropout'])
        
        self.mha_2=tf.keras.layers.MultiHeadAttention(num_heads=config['num_heads'],
                                                    key_dim=config['model_dim'],
                                                    dropout=config['dropout'])
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.feed_forward=FeedForward(config)
   
    def call(self,text_inputs,image_inputs,mask):
        # Obtain the masks
        combined_mask=self.get_combined_mask(mask)
        padding_mask=tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
        # print("Combined mask",combined_mask)
        # print("\n padding mask",padding_mask)
       
        mha_1_out = self.mha_1(query=text_inputs,
                       value=text_inputs,
                       key=text_inputs,
                       attention_mask=combined_mask)
        norm_1_out = self.layer_norm_1(mha_1_out+text_inputs)
        
        mha_2_out = self.mha_2( query=norm_1_out,
                       value=image_inputs,
                       key=image_inputs,
                       attention_mask=padding_mask
                              ) 
        norm_2_out= self.layer_norm_2(mha_2_out+norm_1_out)
        
        ffw_out = self.feed_forward(norm_2_out)
        norm_3_out= self.layer_norm_3(ffw_out+norm_2_out)
        
        return norm_3_out
        
    def get_combined_mask(self,mask):
        input_shape = tf.shape(mask) # (batch_size, seq_len)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        h_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32) # (batch_size,1,seq_len)

        # Generate the causal mask
        i = tf.range(sequence_length)[:,tf.newaxis]
        j = tf.range(sequence_length)
        d_mask = tf.cast(i >= j, dtype="int32") # Fill lower left diagonal
        d_mask = tf.reshape(d_mask, (1, input_shape[1], input_shape[1])) # Add batch dimension (1,seq_len,seq_len)

        # mult=tf.constant([batch_size.numpy(),1,1],dtype=tf.int32) 
        mult = tf.concat([tf.expand_dims(batch_size, -1), 
                          tf.constant([1, 1], 
                          dtype=tf.int32)],
                          axis=0,)
        causal_mask=tf.tile(d_mask,mult) # Tile along batch dimension (batch_size,seq_len,seq_len)

        # Combine causal mask and h_mask
        combined_mask = tf.minimum(h_mask, causal_mask) # (batch_size,seq_len,seq_len)
        return combined_mask

    
class FullDecoder(tf.keras.layers.Layer):
    def __init__(self,config,**kwargs):
        """Make a full decoder with embeddings, transformer decoder and output layers. 
        
        Combines the TextEmbeddings, TransformerDecoder and output layers to 
        form a full decoder that takes in images and texts and outputs probabilities
        of words in the vocabulary. 

        Args: 
            config(dict): Configuration dictionary. 
            training_texts(tf.dataset): Texts to adapt the tokenizer.
            vocabulary_size(int): Vocab size.
        """
        super().__init__(**kwargs)
        self.num_decoder_blocks=config['num_decoder_blocks']
        
        self.text_embeddings=TextEmbeddings(config)
        self.transformer_decoder=[TransformerDecoder(config) 
                                  for _ in range(config['num_decoder_blocks'])]
        self.softmax=tf.keras.layers.Dense(config['vocab_size'],activation='softmax')
        
    def call(self,text_tokens,image_inputs,mask):
        x = self.text_embeddings(text_tokens)
        for i in range(self.num_decoder_blocks):
            x = self.transformer_decoder[i](x,image_inputs,mask)
        probs = self.softmax(x)
        return probs


        
# Code to implement pre-layer normalization transformer: ----------------------

class PreNormTransformerEncoder(tf.keras.layers.Layer):
    """The transformer encoder part without positional encoding. """
    def __init__(self,config,**kwargs):
        super().__init__( **kwargs)
        self.layer_norm_1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha=tf.keras.layers.MultiHeadAttention(
            num_heads=config['num_heads'],
            # The dimensions of the dot product for attention in every head
            key_dim=config['model_dim'], 
            dropout=config['dropout']
        )
        self.ffw=FeedForward(config)
    def call(self,inputs):
        norm_1_out = self.layer_norm_1(inputs)
        mha_out = self.mha(query = norm_1_out,
                           value = norm_1_out,
                           key = norm_1_out)
        add_1_out = (inputs+mha_out)
        norm_2_out = self.layer_norm_2(add_1_out)
        ffw_out = self.ffw(norm_2_out)
        add_2_out = add_1_out + ffw_out
        return(add_2_out)

class PreNormFullEncoder(tf.keras.layers.Layer):
    """"The full encoder containing the image encoder, positional encoder and 
    transformer encoder."""
    def __init__(self,config,**kwargs):
        super().__init__(**kwargs)
        self.num_encoder_blocks=config['num_encoder_blocks']

        self.image_encoder = ImageEncoder(config)
        self.positional_encoder = PositionalEncoder(config,seq_len=100)
        self.transformer_encoders = [PreNormTransformerEncoder(config) 
                                     for _ in range(self.num_encoder_blocks)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    def call(self,inputs):
        x = self.image_encoder(inputs)
        x = self.positional_encoder(x)
        for i in range(self.num_encoder_blocks):
            x=self.transformer_encoders[i](x)
        x = self.norm(x)
        return x

class PreNormTransformerDecoder(tf.keras.layers.Layer):
    def __init__(self,config, **kwargs):
        super().__init__(**kwargs)
        self.mha_1=tf.keras.layers.MultiHeadAttention(num_heads=config['num_heads'],
                                                    key_dim=config['model_dim'],
                                                    dropout=config['dropout'])
        
        self.mha_2=tf.keras.layers.MultiHeadAttention(num_heads=config['num_heads'],
                                                    key_dim=config['model_dim'],
                                                    dropout=config['dropout'])
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.feed_forward=FeedForward(config)
   
    def call(self,text_inputs,image_inputs,mask):
        # Obtain the masks
        combined_mask=self.get_combined_mask(mask)
        padding_mask=tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)

        norm_1_out=self.layer_norm_1(text_inputs)
        mha_1_out = self.mha_1(query=norm_1_out,
                               value=norm_1_out,
                               key=norm_1_out,
                               attention_mask=combined_mask)
        add_1_out = (text_inputs+mha_1_out)  
        norm_2_out = self.layer_norm_2(add_1_out) 
        mha_2_out = self.mha_2( query=norm_2_out,
                                value=image_inputs,
                                key=image_inputs,
                               attention_mask=padding_mask
                              ) 
        add_2_out = (mha_2_out+add_1_out)
        norm_3_out= self.layer_norm_3(add_2_out)   
        ffw_out = self.feed_forward(norm_3_out)
        add_3_out = ffw_out+add_2_out
        return add_3_out
    
    def get_combined_mask(self,mask):
        input_shape = tf.shape(mask) # (batch_size, seq_len)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        h_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32) # (batch_size,1,seq_len)

        # Generate the causal mask
        i = tf.range(sequence_length)[:,tf.newaxis]
        j = tf.range(sequence_length)
        d_mask = tf.cast(i >= j, dtype="int32") # Fill lower left diagonal
        d_mask = tf.reshape(d_mask, (1, input_shape[1], input_shape[1])) # Add batch dimension (1,seq_len,seq_len)

        # mult=tf.constant([batch_size.numpy(),1,1],dtype=tf.int32) 
        mult = tf.concat([tf.expand_dims(batch_size, -1), 
                          tf.constant([1, 1], 
                          dtype=tf.int32)],
                          axis=0,)
        causal_mask=tf.tile(d_mask,mult) # Tile along batch dimension (batch_size,seq_len,seq_len)

        # Combine causal mask and h_mask
        combined_mask = tf.minimum(h_mask, causal_mask) # (batch_size,seq_len,seq_len)
        return combined_mask
    
class PreNormFullDecoder(tf.keras.layers.Layer):
    def __init__(self,config,**kwargs):
        """Build a full decoder with embeddings, transformer decoder and output layer. 
        
        Combines the TextEmbeddings, TransformerDecoder and output layers to 
        form a full decoder that takes in images and texts and outputs probabilities
        of words in the vocabulary. 

        Args: 
            config(dict): Configuration dictionary. 
            training_texts(tf.dataset): Texts to adapt the tokenizer.
            vocabulary_size(int): Vocab size.
        """
        super().__init__(**kwargs)
        self.num_decoder_blocks=config['num_decoder_blocks']
        
        self.text_embeddings=TextEmbeddings(config)
        self.transformer_decoder=[PreNormTransformerDecoder(config) 
                                  for _ in range(config['num_decoder_blocks'])]
        self.norm=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.softmax=tf.keras.layers.Dense(config['vocab_size'],activation='softmax')
        
    def call(self,text_tokens,image_inputs,mask):
        x = self.text_embeddings(text_tokens)
        for i in range(self.num_decoder_blocks):
            x = self.transformer_decoder[i](x,image_inputs,mask)
        x = self.norm(x)
        probs = self.softmax(x)
        return probs
    
class CaptioningTransformer(tf.keras.Model):
    def __init__(self,config,**kwargs):
        super().__init__(**kwargs)
        self.config=config
        self.vocabulary_size=config['vocab_size']
        self.capts_per_img=config['capts_per_img']
        
        if self.config['normalization']=='pre':
        # If prenormalization architecture is indicated:
            self.encoder=PreNormFullEncoder(config)
            self.decoder=PreNormFullDecoder(config)
        else:
        # Fall back to default post-normalization architecture. 
            self.encoder=FullEncoder(config)
            self.decoder=FullDecoder(config)
        
        
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")
        
    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]
    
    def train_step(self,inputs):
        # Unpack the data..
        tokens=inputs['tokens']
        images=inputs['image']
        
        # Initialize batch_loss and batch_accuracy
        batch_loss=0
        batch_acc=0
        
        # Iterate over all captions for an image and do a forward
        # pass for each. 
        for i in range(self.capts_per_img):
        # For every caption corresponding to the same image:
            with tf.GradientTape() as tape:
                # Forward pass-------------------------------------
                y_true, pred_tokens=tokens[:,i,1:],tokens[:,i,0:-1] # split predictors and targets
                mask=tf.math.not_equal(y_true,0) # Build mask for padding tokens.
                image_inputs=self.encoder(images) # Run image through encoder
                y_pred=self.decoder(pred_tokens,image_inputs,mask) # Decode next word probabilities

                # Compute the loss value
                loss = self.calculate_loss(y_true,y_pred,mask)
                batch_loss+=loss
                
            # Backward pass-------------------------------
            # Compute gradients
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
                
            # Compute other metrics--------------------
            acc = self.calculate_accuracy(y_true,y_pred,mask)
            batch_acc+=acc

        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc/self.capts_per_img)
        
        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(), 'acc':self.acc_tracker.result()}
    
    def test_step(self,inputs):
        # Unpack the data..
        tokens=inputs['tokens']
        images=inputs['image']
        
        # Initialize batch_loss and batch_accuracy
        batch_loss=0
        batch_acc=0
        
        # Iterate over all captions for an image and do a forward
        # pass for each. 
        for i in range(self.capts_per_img):
        # For every caption corresponding to the same image:
            
            # Forward pass-------------------------------------
            image_inputs=self.encoder(images) # Run image through encoder
            
            y_true, pred_tokens=tokens[:,i,1:],tokens[:,i,0:-1] # split predictors and targets
            mask=tf.math.not_equal(y_true,0) # Build mask for padding tokens.
           
            y_pred=self.decoder(pred_tokens,image_inputs,mask) # Decode next word probabilities
            
            # Compute the loss value
            loss = self.calculate_loss(y_true,y_pred,mask)
            batch_loss+=loss
                
            # Compute other metrics--------------------
            acc = self.calculate_accuracy(y_true,y_pred,mask)
            batch_acc+=acc

        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(batch_loss)
        self.acc_tracker.update_state(batch_acc/self.capts_per_img)
        
        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result(), 'acc':self.acc_tracker.result()}
    
    def calculate_probs(self,images,pred_tokens,mask):
        image_inputs=self.encoder(images) # Run image through encoder
        y_pred=self.decoder(pred_tokens,image_inputs,mask) # Decode next word probabilities
        return y_pred
        
        
    def calculate_loss(self, y_true, y_pred, mask):
        """Calculates the loss and masks it.
        
        Args:
            y_true (tensor): Batch of tokens with dimension (None,seq_len-1)
                (-1 because of the left shifting of the input sequence)
            y_pred (): A tensor of probabilities for the vocabulary distribution
                it commonly has dimensions (batch_size, seq_len, vocab_size). 
        
        """
        # Get sparse categorical crossentropy loss
        # y_pred has dimensions (batch, seq_len,vocab_size)
        # y_true has dimensions (batch, seq_len)
        loss = self.loss(y_true, y_pred) # (the loss function is configured in `compile()` )
        mask = tf.cast(mask, dtype=loss.dtype) # Cast mask from boolean to float.
        loss *= mask # Truncate loss where there are paddings
        loss= (tf.reduce_sum(loss) / tf.reduce_sum(mask)) # Get average loss for all non-pad positions.
        # print(loss)
        return loss
    
    def calculate_accuracy(self, y_true, y_pred, mask):
        """Calculates masked accuracy.
        
        Args: 
            y_true (Tensor): A tensor of probabilities for the vocabulary distribution
                it commonly has dimensions (batch_size, seq_len, vocab_size). 

        """

        # Find matches between argmax of probabilities and y_true. 
        # argmax yields the index of the highest probability for next word. 
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2)) 
        # From the obtained matches we mask out those that belong to padding positions. 
        accuracy = tf.math.logical_and(mask, accuracy)
        # Convert from boolean to float. 
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        # Divide the total number of matches (not in padding positions). 
        # over the total number of words that are not in padding positions. 
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)