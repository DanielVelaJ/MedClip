

   
Getting Started
===============
Here are all the steps you need to follow before being able to run experiments. 

Cloning the Repository
----------------------

Begin by cloning the repository 

.. code-block:: console

   $ git clone https://github.com/DanielVelaJ/MedClip.git

The directory only has a src folder containing all modules and scripts. 

Preparing Virtual Environment
-----------------------------

In order to install all dependencies, it is useful to make a virtual environment.
For this navigate to the MedClip folder and run the following code to create a 
virtual environment named "venv": 

.. code-block:: console

   $ python -m venv venv
   
This will create a virtual environment in the venv directory. To activate it run:

.. code-block:: console

   $ source venv/bin/activate

Upon succesful activation the terminal will display the name of the virtual environment
as follows:

.. code-block:: console

   (venv) $ 

Now the only step left is to install all dependencies. Navigate to the :file:`Medclip/` 
and install dependencies through the following command: 

.. code-block:: console

   (venv) $ pip install -r requirements.txt

It may also be helpful to create a kernel for jupyter-notebooks as follows: 

.. code-block:: console

   (venv) $ python -m ipykernel install --user --name=myenv

The following sections will take you through the workflow of downloading and 
preparing the datasets. 


   
Download Raw Datasets
---------------------
The first step in the captioining journey begins by downloading the raw datasets
we will use. For this, we created the  :py:mod:`download_data.py <../download_data>` 
script which in turn uses many of the functions provided in the 
:py:mod:`download module <download>` to download different medical image datasets. 
Each dataset is decribed :doc:`here <../datasets>` . 

To run the script run the following lines from the :file:`Medclip/src/` directory: 

.. code-block:: console

   $ python download_data.py

The script will ask which datasets to download and download them accordingly. 
After succesfully running the script, you will notice that a :code:`data` folder 
as been created as well as a :code:`data/raw` directory inside it containing 
the downloaded datasets as follows.

.. code-block:: console

   .
   └── Medclip
       ├── data
       │   └── raw
       │       ├── medpix
       │       └── mimics
       │       └── ...
       └── src

Each folder inside the :code:`data` folder 
contains the raw information of each dataset, namely 
images, .txt and .csv files that will be used when running the 
:py:mod:`prepare_data.py <prepare_data>` script later on. 

.. _preprocess_raw_datasets:

Preprocess Raw Datasets
-------------------------
In this step, we prepare all the data to be ingested by our :py:func:`pipelines.build_pipeline` function in the future.
To do so, we will use the :py:mod:`prepare_data.py <prepare_data>` script. 
This script will take our raw datasets at :file:´MedClip/data/raw´ and convert 
them into **intermediate datasets** to be saved at :file:´MedClip/data/intermediate´

The :py:mod:`prepare_data.py <prepare_data>` script prompts the user on which datasets to prepare. 
Note that in order to prepare a dataset, it must be already downloaded in the 
:file:`Medclip/data/raw` directory. To execute the script simply run:

.. code-block:: console

   (venv) $ python prepare_data.py 

Now we have a clean version of the raw datasets which is only a .csv file pointing
to image paths and other columns containing captions. For more info about intermediate datasets
look at :py:mod:`prepare_data.py <prepare_data>`

