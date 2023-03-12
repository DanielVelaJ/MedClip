.. role:: python(code)
  :language: python
  :class: highlight
  
Using Models
============
This library currently contains only one kind of model, this is the 
:py:class:`CaptioninTransformer <models.CaptioningTransformers.CaptioningTransformer>` 
which in turn works with the help of a :py:class:`tokenizers.KerasTokenizer`.

The model uses tokenized data to train and at the moment of prediction, it 
predicts output tokens. Therefore, a few steps are required before we can 
start training. 

Building a captioning pipeline
------------------------------
To build a captioning pipeline we will use the :py:mod:`pipelines` module. This
module contains functions to create pipelines from our 
:ref:`intermediate datasets <preprocess_raw_datasets>`. Let's use 
:py:func:`pipelines.build_pipeline`. 

.. doctest::
   :skipif: True
   
   >>> from pipelines import build_pipeline 
   >>> pipeline=build_pipeline()
   >>> pipeline.keys()
   dict_keys(['captioning'])

The pipeline is a dictionary containing one key :code:`'captioning'`. Let's 
inspect it: 

.. doctest::
   :skipif: True
   
   >>> pipeline['captioning'].keys()
   dict_keys(['train', 'val', 'test', 'train_captions'])

We can see three elements now. :python:`pipeline['captioning']['train']`