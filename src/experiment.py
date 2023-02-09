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
import seaborn as sns
# Tokenizers
from tokenizers import KerasTokenizer
# Model
from models.CaptioningTransformers import CaptioningTransformer
from models.Callbacks import CaptioningCallback
from models.model_utils import load_captioning_model
# Results
from prediction import Predict
from evaluation import Evaluate
import nvidia_smi
import gc
import multiprocessing
from multiprocessing import Process, Queue




# Reset Keras Session

  

class Experiment(object):
    def __init__(self,config,save_path,log_dir):
        self.config=config
        self.save_path=save_path
        self.log_dir=log_dir
        self.tokenizer=None
        self.metric=config['metric']
        
    def init_pipeline(self):
        """ Load data pipeline to self.data"""
        if self.config['dataset']=='medpix':
            self.data = build_pipeline()
        elif self.config['dataset']=='coco':
            self.data = build_coco_pipeline() 
        
        
        
    def init_tokenizer(self):
        """Initialize tokenizer and build vocabulary."""
        
        # Prepare tokenizer data
        tokenizer_data=self.data['captioning']['train_captions']
        # Initialize tokenizer
        tokenizer=KerasTokenizer(seq_len=self.config['seq_len'])
        tokenizer.build_vocabulary(tokenizer_data) # Build vocabulary
        # Save vocabulary to disk
        tokenizer.save_vocabulary(self.save_path)
        
        # Save vocabulary size to config dictionary
        self.config['vocab_size']=tokenizer.vocab_size
        self.tokenizer=tokenizer
        
        # Save experiment configuration
        os.makedirs(self.save_path,exist_ok=True)
        config_path=self.save_path+'/'+self.save_path.split('/')[-1]+'.config'
        
        with open(config_path, 'wb') as fp:
            pickle.dump(self.config, fp)
            
        with open(config_path+'_txt', 'w') as fp:
            fp.write(json.dumps(self.config, indent=4))
        return
        
       
        
    def tokenize_data(self):
        """Add tokenizing step to pipeline"""
        
        def tokenize(ds_row):
            tokens=self.tokenizer.tokenize(ds_row['text'])
            return {'image':ds_row['image'],
                   'text':ds_row['text'],
                   'tokens':tokens}
        batch_size=32
        # Obtain data for training, validation and test 
        train_data = (self.data['captioning']['train']
                      .map(tokenize,num_parallel_calls=tf.data.AUTOTUNE)
                      .batch(batch_size)
                      # .take(1)
                     )
        val_data = (self.data['captioning']['val']
                    .map(tokenize,num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size)
                    # .take(1)
                   )
        test_data = (self.data['captioning']['test']
                     .map(tokenize,num_parallel_calls=tf.data.AUTOTUNE)
                     # .take(1)
                    )
        
        self.train_data=train_data
        self.val_data=val_data
        self.test_data=test_data
        return
        
    def initialize_callbacks(self):
        """Initialize the commonly used callbacks."""
        if self.metric=='val_acc':
            monitor='val_acc'
            mode='max'
            
        if self.metric=='val_loss':
            monitor='val_loss'
            mode='min'

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint( # Callback to save the model every epoch end
                filepath=self.save_path+'/'+self.save_path.split('/')[-1],
                save_weights_only=True,
                monitor=monitor,
                mode=mode,
                save_best_only=True)

        tensorboard_callback=tf.keras.callbacks.TensorBoard(  # Callback to log loss and accuracy every epoch
            log_dir=self.log_dir,
            histogram_freq=None,
            write_graph=True,
            write_images=False,
            write_steps_per_second=False,
            update_freq='epoch',
            profile_batch=0,
            embeddings_freq=1,
            embeddings_metadata=None)

        # monitor: Quantity to be monitored.
        # min_delta: Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute 
        #            change of less than min_delta, will count as no improvement.
        # patience: Number of epochs with no improvement after which training will be stopped.
        # verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays messages when the callback takes an action.
        # mode: One of {"auto", "min", "max"}. In min mode, training will stop when the quantity monitored has stopped 
        #       decreasing; in "max" mode it will stop when the quantity monitored has stopped increasing; in "auto" 
        #       mode, the direction is automatically inferred from the name of the monitored quantity.
        # baseline: Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement
        #           over the baseline.
        # restore_best_weights: Whether to restore model weights from the epoch with the best value of the monitored 
        #                       quantity. If False, the model weights obtained at the last step of training are used.
        #                       An epoch will be restored regardless of the performance relative to the baseline. If 
        #                       no epoch improves on baseline, training will run for patience epochs and restore weights 
        #                       from the best epoch in that set.
        # start_from_epoch: Number of epochs to wait before starting to monitor improvement. This allows for a warm-up 
        #                   period in which no improvement is expected and thus training will not be stopped.

        early_stopping_callback=tf.keras.callbacks.EarlyStopping(
                                                                    monitor=monitor,
                                                                    min_delta=.0001,
                                                                    patience=6,
                                                                    verbose=1,
                                                                    mode=mode,
                                                                    baseline=None,
                                                                    restore_best_weights=True,
                                                                 )
        captioning_callback= CaptioningCallback(config=self.config,
                                       log_dir=self.log_dir)
        
        self.callbacks=[checkpoint_callback,tensorboard_callback,
                        early_stopping_callback,captioning_callback]

        return 
    
    def create_and_train_model(self):
        """Create the model and train it."""
        
        tf.keras.backend.clear_session()
        self.model=CaptioningTransformer(self.config)
        loss=tf.keras.losses.SparseCategoricalCrossentropy(reduction='none',from_logits=False)
        opt = tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        # opt = keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(loss=loss, optimizer=opt)

        self.history=self.model.fit(self.train_data,
                                    validation_data=self.val_data,
                                    epochs=self.config['max_epochs'],
                                    callbacks=self.callbacks)
        
        return
    
    def plot_learning_curves(self):
        fig_path=self.save_path+'/'+self.save_path.split('/')[-1]+'_xx'
        
        figure=plt.figure(figsize=(9,6),dpi=100)
        epochs=len(self.history.history['loss'])
        # xticks=range(epochs)
        
        plt.plot(self.history.history['loss'],label='train')
        plt.plot(self.history.history['val_loss'],label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        # plt.xticks(xticks)
        plt.tight_layout()
        figure.savefig(fig_path.replace('xx','loss.png'))

        figure=plt.figure(figsize=(9,6),dpi=100)
        plt.plot(self.history.history['acc'],label='train')
        plt.plot(self.history.history['val_acc'],label='validation')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        # plt.xticks(xticks)
        plt.tight_layout()
        figure.savefig(fig_path.replace('xx','accuracy.png'))
        
    
    def make_predictions(self):
        """Make and save predictions over the test set."""
        Predict.dataset_captions(self.test_data,
                                 self.model,
                                 self.tokenizer,
                                 model_path=self.save_path
                                )
        
    def evaluate_model(self):
        evaluate=Evaluate(self.save_path)
        evaluate.evaluate_all()
        self.results = evaluate.results
        pass
    
    def run(self):
        print('Running experiment with the following configuration:')
        print(self.config)
        print('Loading data pipeline...')
        self.init_pipeline()
        print('Initializing tokenizer...')
        self.init_tokenizer()
        print('Tokenizing data...')
        self.tokenize_data()
        print('Instantiating callbacks...')
        self.initialize_callbacks()
        print('Creatng and training model...')
        self.create_and_train_model()
        print('Plotting learning curves...')
        self.plot_learning_curves()
        print('Making predictions...')
        self.make_predictions()
        print('Evaluating...')
        self.evaluate_model()
        print('Clearing GPU memory')
        tf.keras.backend.clear_session()
        del self.model
        del self.tokenizer
        del self.train_data
        del self.val_data
        del self.test_data
        tf.keras.backend.clear_session()
        gc.collect()
        
#     def run(self):
#         p = multiprocessing.Process(target=self.run_w)
#         p.start()
#         p.join()
        

        
class ExperimentCluster(object):
    def __init__(self, configs, cluster_path): 
        self.configs = configs
        self.cluster_path = cluster_path
        self.results={}
    
    def run(self): 
        for i,config in enumerate(self.configs):
            experiment_path=self.cluster_path+'/'+str(i)
            log_dir = self.cluster_path+'/logs/'+str(i)
            
             # Run the experiment in a subprocess.
            q = Queue() # Create queue to recover arguments
            p = Process(target=self.run_experiment, 
                        args=(q,config,experiment_path,log_dir)) # Create subprocess
            p.start() # Launch process
            results = q.get() # Read queue for results. 
            p.join() # End process?
            
            # Retrieve the results from this experiment run: 
            if i==0:
            # If this is the first experiment: 
                for key,value in results.items():
                    self.results[key]=value
                    
        
            else:
                for key,value in results.items():
                    self.results[key]+=value
 
                    
            self.save_results()

            
    
    def run_experiment(self,q,config,save_path,log_dir):
        """ Runs experiment in subprocess. """
        # Initialize an experiment
        experiment= Experiment(config,
                                   save_path=save_path,
                                   log_dir=log_dir)
        experiment.run() # run the experiment 
        q.put(experiment.results) # write results to queue
        
    def save_results(self):
        self.cluster_results_path = self.cluster_path+'/'+'cluster_results.csv'
        pd.DataFrame(self.results).to_csv(self.cluster_results_path)
        
        
        
        
        

class AblationStudy(object):
    def __init__(self,ablation_config,study_path):
        super().__init__()
        self.ablation_config=ablation_config
        self.study_path = study_path
        self.results={}
        
         # Save study configuration
        os.makedirs(self.study_path,exist_ok=True)
        self.ab_config_path=(self.study_path+'/'+self.study_path.split('/')[-1]
                            +'.ab_config')
        
        with open(self.ab_config_path, 'wb') as fp:
            pickle.dump(self.ablation_config, fp)
            
        with open(self.ab_config_path+'_txt', 'w') as fp:
            fp.write(json.dumps(self.ablation_config, indent=4))
        return
    
    def get_configs(self):
        """Obtain a list of configurations for the ablation study."""
        
        # The default config is made using first value of every parameter list. 
        default_config={key:value[0] for (key,value) in 
                        self.ablation_config.items()}
        self.configs=[default_config] # Add default config to configs. 
        
        for param,values in self.ablation_config.items():
        # For every parameter
            if len(values)>1:
            # If there is other than the default parameter. 
                for value in values[1:]:
                # Get configurations with the default config + the modified 
                # parameter.
                    config=default_config.copy()
                    config[param]=value
                    self.configs.append(config)
   
    def run(self):
        # Obtain list of configurations for the ablation study
        self.get_configs()
        
        # Make a foder to save all experiments from this ablation study
        os.makedirs(self.study_path,exist_ok=True)
        for i,config in enumerate(self.configs):
        # For each configuration
        
            save_path = self.study_path+'/'+str(i)
            log_dir = self.study_path+'/logs/'+str(i)
            
            # Run the experiment in a subprocess.
            q = Queue() # Create queue to recover arguments
            p = Process(target=self.run_experiment, 
                        args=(q,config,save_path,log_dir)) # Create subprocess
            p.start() # Launch process
            results = q.get() # Read queue for results. 
            p.join() # End process?
            
            # Retrieve the results from this experiment run: 
            if i==0:
            # If this is the first experiment: 
                for key,value in results.items():
                    self.results[key]=value
        
            else:
                for key,value in results.items():
                    self.results[key]+=value
 
                    
            self.save_results()
            self.plot_ablation_results()
    
    def run_experiment(self,q,config,save_path,log_dir):
        """ Runs experiment in subprocess. """
        # Initialize an experiment
        experiment= Experiment(config,
                                   save_path=save_path,
                                   log_dir=log_dir)
        experiment.run() # run the experiment 
        q.put(experiment.results) # write results to queue
        
            


        
    
    def save_results(self):
        self.compiled_results_path = self.study_path+'/'+'compiled_results.csv'
        pd.DataFrame(self.results).to_csv(self.compiled_results_path)
    
    def plot_ablation_results(self):
        # Open the results dataframe
        df=pd.read_csv(self.compiled_results_path)
        
        # Obtain the colum names of parameters that were explored in the ablation study. 
        explored_params=[key 
                         for (key,value) in self.ablation_config.items() 
                         if len(value)>1]

        # Obtain the default configuration for all parameters 
        default_config= {key:value[0] 
                         for (key,value) in self.ablation_config.items()}

        # For every explored parameter, make a sub_dataframe where all parameters 
        # are fixed to the default except for the explored parameter. 
        for param in explored_params:
        #For every explored parameter: 
            # Substract the parameter from the default configuration. 
            default_config_sub = {key:value for (key,value) in default_config.items() 
                                  if key != param}

            # Make a filter for the dataframe based on this sub_configuration
            # this configuration is fixed for all parameter in their default config
            # except for the explored parameter. 

            for i,(key,value) in enumerate(default_config_sub.items()): 
                if df[key].dtype=='O':
                    value=str(value)

                if i==0: 
                    filt = (df[key]==value)
                else:
                    filt = filt & (df[key]==value)
            
            # Filter the data according to the constructed filter.
            sub_df = df.loc[filt]
            
            # Initialize a path to save figures. 
            # 'xx' will be replaced by the name of the specific figure
            # when the saving command is called. 
            fig_path=self.study_path+'/'+param+'_vs_xx'
            # Compare bleu scores in a figure and save the figure.
            bleu_figure,ax=plt.subplots(1,1)
            sns.barplot(data=sub_df, x='bleu', y=param,ax=ax,
                        orient='h',width=0.2,color='#2589BD')
            plt.tight_layout()
            bleu_figure.savefig(fig_path.replace('xx','bleu.png'))


            # Compare rouge scores in a figure and save the figure.
            melt_sub_df=pd.melt(sub_df, id_vars=[param],value_vars=['rougeL_f_low','rougeL_f_mid','rougeL_f_high'])
            melt_sub_df=melt_sub_df.replace('rougeL_f_low','low')
            melt_sub_df=melt_sub_df.replace('rougeL_f_mid','mid')
            melt_sub_df=melt_sub_df.replace('rougeL_f_high','high')


            rouge_figure,ax=plt.subplots(1,1)

            sns.barplot(data=melt_sub_df,
                        orient='h',
                        x='value', 
                        y=param,
                        hue='variable',
                        width=0.4,
                        ax=ax,
                        # color='red'
                        palette=['#FF7C70','#FF1F0A','#A30E00']
                       )
            ax.legend().set_title('')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            ax.set_xlabel('Rouge-L f-measure')
            plt.tight_layout()
            rouge_figure.savefig(fig_path.replace('xx','rouge.png'))
            
            
    