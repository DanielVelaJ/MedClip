

# MedClip     


This is the official repository for the MedClip research project from [MindKind research group](https://mindkindgroup.com).
<img src="https://mindkindgroup.com/wp-content/uploads/2021/05/logo-negro.svg" alt="Mindkind logo" width="400"/>

The project investigates and compares different pretraining tasks for medical image feature extraction and captioning. 

## Workflow
The experiments are structured as follows: 

download datasets ---> prepare data ---> build models ---> run experiments

Each stage is run by a python script of its own that allows to custumize options in every step.																		
## Downloading datasets
To download datasets use the download_all.py script from the src directory. This folder will download and extract the zip files for each dataset and sort their contents into the data/raw directory. 

```bash
python download_all.py
```

## Preparing data
The downloaded data is used to produce clean dataframes aswell as model training material. Each downloaded dataset generates a dataframe as shown below
|File|Modality|Anatomy|Patient history| Findings | Impression| Diagnosis
|--|--|--|--|--|--|--|
| path to file | imaging modality | imaged anatomy | clinicla history of patient in natural language| findings in natural language| diagnostic impression | concise diagnosis



For some datasets there may be extra columns or missing columns but the names are consistent across all generated datasets. To prepare the data run the prepare_data.py script from the src directory:
```python
python prepare_data.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)
