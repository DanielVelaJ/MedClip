from pipelines import build_pipeline, build_coco_pipeline
from experiment import Experiment, AblationStudy, ExperimentCluster

# RUNNING A SINGLE EXPERIMENT---------------------------------------------------
# data = build_pipeline()
# # Setup configuration
# config={
#     'database':'medpix',
#     'model_name':'exp_3',
#     'seq_len':178,
#     'num_heads':3,
#     'model_dim':252,
#     'dropout':0.2,
#     'img_backbone_trainable':False,
#     'img_size':(299,299,3),
#     'img_backbone':'efficientnetb0',
#     'intermediate_size':250*4,
#     'vocab_size':None,
#     'capts_per_img':1,
#     'num_encoder_blocks':3,
#     'num_decoder_blocks':3,
#     'max_epochs':100,
#     'learning_rate':0.001,
#     'normalization':'pre'
# }

# experiment=Experiment(config,data,'../models/medpix_250_lr001_acc_pre',
#                       '../model_logs/medpix_250_acc_lr001_pre',
#                      metric='val_acc')
# experiment.run()


# RUNNING AN ABLATION STUDY-----------------------------------------------------
ablation_config={
    'dataset':['medpix'],
    'seq_len':[178],
    'num_heads':[16,8,4],
    'model_dim':[512,256,128],
    'dropout':[0.2,0.4,0.6],
    'img_backbone_trainable':[False],
    'img_size':[(299,299,3)],
    'img_backbone':['efficientnetb0'],
    'intermediate_size':[4*512],
    'capts_per_img':[1],
    'num_encoder_blocks':[3],
    'num_decoder_blocks':[3],
    'max_epochs':[100],
    'learning_rate':[0.001],
    'metric':['val_acc'],
    'normalization':['pre']
}
ablation_study=AblationStudy(ablation_config,'../models/study_1')
ablation_study.run()


# RUNNING AN EXPERIMENT CLUSTER-------------------------------------------------
# Setup 3 different configs: 
# config_1={
#     'name':'model_big',
#     'dataset':'medpix',
#     'model_name':'exp_3',
#     'seq_len':178,
#     'num_heads':8,
#     'model_dim':512,
#     'dropout':0.2,
#     'img_backbone_trainable':False,
#     'img_size':(299,299,3),
#     'img_backbone':'efficientnetb0',
#     'intermediate_size':512*4,
#     'vocab_size':None,
#     'capts_per_img':1,
#     'num_encoder_blocks':6,
#     'num_decoder_blocks':6,
#     'max_epochs':2,
#     'learning_rate':0.001,
#     'normalization':'pre',
#     'metric':'val_acc'
# }

# config_2={
#     'name':'model_mid',
#     'dataset':'medpix',
#     'model_name':'exp_3',
#     'seq_len':178,
#     'num_heads':4,
#     'model_dim':256,
#     'dropout':0.2,
#     'img_backbone_trainable':False,
#     'img_size':(299,299,3),
#     'img_backbone':'efficientnetb0',
#     'intermediate_size':256*4,
#     'vocab_size':None,
#     'capts_per_img':1,
#     'num_encoder_blocks':3,
#     'num_decoder_blocks':3,
#     'max_epochs':2,
#     'learning_rate':0.001,
#     'normalization':'pre',
#     'metric':'val_acc'
# }


# config_3={
#     'name':'model_small',
#     'dataset':'medpix',
#     'model_name':'exp_3',
#     'seq_len':178,
#     'num_heads':1,
#     'model_dim':128,
#     'dropout':0.2,
#     'img_backbone_trainable':False,
#     'img_size':(299,299,3),
#     'img_backbone':'efficientnetb0',
#     'intermediate_size':128*4,
#     'vocab_size':None,
#     'capts_per_img':1,
#     'num_encoder_blocks':1,
#     'num_decoder_blocks':1,
#     'max_epochs':2,
#     'learning_rate':0.001,
#     'normalization':'pre',
#     'metric':'val_acc'
# }

# configs=[config_1,config_2,config_3]
# experiment_cluster=ExperimentCluster(configs,'../models/cluster_1')
# experiment_cluster.run()
