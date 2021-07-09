# Code for AGQA baselines

This repository contains code for our baselines HCRN, HME, and PSAC. For all three baselines, we used their code for the FrameQA questions in the TGIF-QA benchmark, then adjusted to fit our data structure. These models are the baselines used for Action Genome Question Answering (AGQA). Check out our paper [here](https://arxiv.org/pdf/2103.16002.pdf), and our benchmark data [here](https://cs.stanford.edu/people/ranjaykrishna/agqa/). 

```
@inproceedings{GrundeMcLaughlin2021AGQA,
title={AGQA: A Benchmark for Compositional Spatio-Temporal Reasoning},
author={Grunde-McLaughlin, Madeleine and Krishna, Ranjay and Agrawala, Maneesh},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2021}
}
```

# Data

## Appearance Features

We shared appearance features across models for consistency (RESNET for appearance and RESNEXT for accuracy). The format of those features differs across models, so we include 4 visual features files. Find the visual features stored [here](https://drive.google.com/drive/u/0/folders/1c51H6rxGHcc8en_sboFxH7Oqb_SVb2oQ).

* tgif-qa_frameqa_appearance_feat.h5 (10 GB)
* tgif-qa_frameqa_motion_feat.h5 (630 MB)
* TGIF_RESNET_pool5.hdf5 (2.75 GB) 
* TGIF_RESNEXT.hdf5 (2.75 GB)

## Questions formatted

All three models use a .csv version of the data. Find the questions in a .csv format stored [here](https://drive.google.com/drive/u/0/folders/14GtSnCvM8Is73QcxTsoQe1TuW-5yZfiB).


* Balanced (--metric balanced): 
* Compo (--metric compo): 
* Steps (--metric steps_templ): 

# Models

For each model we’ve included the files with changed code. The files listed below can be found in the ```models/[MODEL]``` directory. Throughout the files, areas that need paths updated are marked with the comment ```TODO: PATH```. HCRN and HME each save multiple versions of the mode. The location specifying which model will be used is marked with the comment ```MODEL CHANGE```.

## HCRN

Find the code and set-up instructions on the [HCRN Github](https://github.com/thaolmk54/hcrn-videoqa)

### Preprocess questions

Train: ```python preprocess/preprocess_questions.py --dataset tgif-qa --question_type frameqa --glove_pt data/glove/glove.840.300d.pkl --mode train --metric [METRIC]```  

Test: ```python preprocess/preprocess_questions.py --dataset tgif-qa --question_type frameqa --mode test --metric [METRIC]```

### Run model

Train: ```python train.py --cfg configs/tgif_qa_frameqa.yml --metric [METRIC]```

Validation: ```python validate.py --cfg configs/tgif_qa_frameqa.yml --metric [METRIC] --mode val```

Test: ```python validate.py --cfg configs/tgif_qa_frameqa.yml --metric [METRIC] --mode test```

### Updated code files
* config.py
* DataLoader.py
* model/HCRN.py
* train.py
* validate.py
* preprocess/preprocess_features.py
* preprocess/preprocess_questions.py
* preproccess/datautils/tgif_qa.py


### Appearance features

* tgif-qa_frameqa_appearance_feat.h5
* tgif-qa_frameqa_motion_feat.h5

### Other adjustments

*TODO: PATH:*
There are paths to be changed in DataLoader.py, train.py, validate.py, preprocess_features.py, preprocess_questions.py.

*MODEL CHANGE:*
Because AGQA is larger than TGIF-QA, we changed the original code to validate more often than every epoch. HCRN originally chose the model with the highest validation score, but we now save both that model, and the current model. Code to switch between these two options is in validate.py, line 106.
   
*Parameters:*
Dropout of 0.25 on the feature aggregation layer and 0.15 elsewhere, and making the weight_decay=1e-3. We found these parameters best reduced overfitting.

*Blind Model:*
In DataLoader.py, on line 85, there is a comment block section with 4 lines of code to uncomment in order to mask the visual input and perform the blind version of the experiment. 


## HME

Find the code and set-up instructions on the [HME Github](https://github.com/fanchenyou/HME-VideoQA)

### Run model

Train: ```python main.py --task=FrameQA --metric=[METRIC]```

Test: ```python main.py --task=FrameQA --metric=[METRIC] --test=1```

### Updated code files

* main.py
* make_tgif.py
* data_util/tgif.py

### Appearance features

* TGIF_RESNET_pool5.hdf5
* tgif-qa_frameqa_motion_feat.h5	            
* TGIF_RESNEXT.hdf5

### Other adjustments

*TODO: PATH:*
There are paths to change in all 3 files.

*MODEL CHANGE:*
The HME code saves models regularly. To choose which saved model to use, change the path on line 134 of main.py.


## PSAC

Find the code and set-up instructions on the [PSAC Github](https://github.com/lixiangpengcs/PSAC) 


### Run model

Train: ```python main_frame.py --metric=[METRIC] --refresh=true```

Test: ```python main_frame.py --metric=[METRIC] --test_phase=True```

### Updated code files

* dataset.py
* models/FrameQA_model.py
* main_frame.py
* train.py

We also changed code so that it would not re-tokenize the data every time we trained. Make these two directories for saving the tokenized data.

* data/tokenized
* data/cache

## Appearance features

* TGIF_RESNET_pool5.hdf5

### Other adjustments

*TODO: PATH:* There are paths to change in all 3 dataset.py and main_frame.py

