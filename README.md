# Code for AGQA baselines

This repository contains code for our baselines HCRN, HME, and PSAC. For all three baselines, we used their code for the FrameQA questions in the TGIF-QA benchmark, then adjusted to fit our data structure. These models are the baselines used for Action Genome Question Answering (AGQA) and AGQA-Decomp. For the AGQA benchmark, check out our paper [here](https://arxiv.org/pdf/2103.16002.pdf), and our benchmark data and updated AGQA 2.0 version [here](https://cs.stanford.edu/people/ranjaykrishna/agqa/). For the AGQA-Decomp benchmark, check out our paper [here](https://arxiv.org/pdf/2204.07190.pdf), and our benchmark data [here](https://agqa-decomp.cs.washington.edu/#data). Additionally, the evaluation scripts for the AGQA-Decomp benchmark can be found in the ```agqa_decomp_evaluation``` folder.

```
@inproceedings{GrundeMcLaughlin2021AGQA,
title={AGQA: A Benchmark for Compositional Spatio-Temporal Reasoning},
author={Grunde-McLaughlin, Madeleine and Krishna, Ranjay and Agrawala, Maneesh},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2021}
}

@inproceedings{Gandhi2022AGQADecomp,
title={Measuring Compositional Consistency for Video Question Answering},
author={Gandhi, Mona* and Gul, Mustafa Omer* and Prakash, Eva and Grunde-McLaughlin, Madeleine and Krishna, Ranjay and Agrawala, Maneesh},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2022}}
```

# Data

## Appearance Features

We shared appearance features across models for consistency (RESNET for appearance and RESNEXT for accuracy). The format of those features differs across models, so we include 4 visual features files. Find the visual features stored [here](https://drive.google.com/drive/u/0/folders/1c51H6rxGHcc8en_sboFxH7Oqb_SVb2oQ). The file names are the same as the original baselines (so they reference tgif-qa). However, these files include the features for the videos used in AGQA.

* tgif-qa_frameqa_appearance_feat.h5 (10 GB)
* tgif-qa_frameqa_motion_feat.h5 (630 MB)
* TGIF_RESNET_pool5.hdf5 (2.75 GB) 
* TGIF_RESNEXT.hdf5 (2.75 GB)

## Questions formatted

All three models use a .csv version of the data. For the AGQA benchmark, find the questions in a .csv format stored [here](https://agqa-decomp.cs.washington.edu/data/agqa2/csvs.zip). For the AGQA-Decomp benchmark, visit [here](https://agqa-decomp.cs.washington.edu/#data) for the questions in a .csv format.


* Balanced (--metric balanced): 
* Compo (--metric compo): 
* Steps (--metric steps_templ): 

# Models

For each model we’ve included the files with changed code. The files listed below can be found in the ```models/[MODEL]``` directory. Throughout the files, areas that need paths updated are marked with the comment ```TODO: PATH```. HCRN and HME each save multiple versions of the model. The location specifying which model will be used is marked with the comment ```MODEL CHANGE```. Additionally, we added new dropout layers for HCRN and PSAC to reduce overfitting. These additions are marked with the comment ```NEW DROPOUT``` in the files. We attempted the same for HME but found that the default model architecture performed better.

## HCRN

Find the code and set-up instructions on the [HCRN Github](https://github.com/thaolmk54/hcrn-videoqa)

### Preprocess questions

Train: ```python preprocess/preprocess_questions.py --dataset tgif-qa --question_type frameqa --glove_pt data/glove/glove.840.300d.pkl --mode train --metric [METRIC]```  

Test: ```python preprocess/preprocess_questions.py --dataset tgif-qa --question_type frameqa --mode test --metric [METRIC]```

### Run model

Train: ```python train.py --cfg configs/tgif_qa_frameqa_[BENCHMARK].yml --metric [METRIC]```

Validation: ```python validate.py --cfg configs/tgif_qa_frameqa_[BENCHMARK].yml --metric [METRIC] --mode val```

Test: ```python validate.py --cfg configs/tgif_qa_frameqa_[BENCHMARK].yml --metric [METRIC] --mode test```

### Updated and added code files
* config.py
* configs/tgif_qa_frameqa_agqa.py
* configs/tgif_qa_frameqa_decomp.py
* DataLoader.py
* model/HCRN.py
* model/CRN.py
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
Because AGQA is larger than TGIF-QA, we changed the original code to validate more often than every epoch. HCRN originally chose the model with the highest validation score, but we now save both that model, and the current model. Code to switch between these two options is in validate.py, line 107.
   
*NEW DROPOUT:*
In order to reduce overfitting, we added new dropout layers to HCRN.py and CRN.py. The additions are marked with ```NEW DROPOUT```

*Parameters:*
We used a dropout probability of 0.25 on the feature aggregation layer and 0.15 elsewhere, and made the weight_decay=5e-4. We set lr=1.7e-4 and lr=1.6e-4 for AGQA and AGQA-Decomp respectively. We found these parameters best reduced overfitting.

*Blind Model:*
In DataLoader.py, on line 85, there is a comment block section with 4 lines of code to uncomment in order to mask the visual input and perform the blind version of the experiment. 


## HME

Find the code and set-up instructions on the [HME Github](https://github.com/fanchenyou/HME-VideoQA)

### Run model

Train: ```python main.py --task=FrameQA --metric=[METRIC]```

Test: ```python main.py --task=FrameQA --metric=[METRIC] --test=1```

### Updated and added code files

* main.py
* make_tgif.py
* data_util/tgif.py
* split_csv.py

### Appearance features

* TGIF_RESNET_pool5.hdf5
* tgif-qa_frameqa_motion_feat.h5	            
* TGIF_RESNEXT.hdf5

### Other adjustments

*TODO: PATH:*
There are paths to change in all 3 files.

*MODEL CHANGE:*
The HME code saves models regularly. To choose which saved model to use, change the path on line 134 of main.py.

*CSV SPLIT:*
Due to HME's slow evaluation speed, we added the option to split the test set into four smaller .csv files and evaluate in a parallelized manner. To split the test .csv, run ```python split_csv.py --metric [METRIC]``` in the folder holding the .csv files. To evaluate on one of these split .csv files, run ```python main.py --task=FrameQA --metric=[METRIC] --test=1 --csv_split=[SPLIT_IDX]``` This will generate a prediction .json file specific to this split. The default value of --csv_split is 0 and will make the model evaluate on the entire test set.


## PSAC

Find the code and set-up instructions on the [PSAC Github](https://github.com/lixiangpengcs/PSAC) 


### Run model

Train/val split: ```python split_csv.py --metric=[METRIC]```

Train: ```python main_frame.py --metric=[METRIC]```

Test: ```python main_frame.py --metric=[METRIC] --test_phase=True```

### Updated and added code files

* dataset.py
* models/FrameQA_model.py
* models/language_model.py
* models/model_utils.py
* models/classifier.py
* main_frame.py
* train.py
* split_csv.py

We also changed code so that it would not re-tokenize the data every time we trained. Make these two directories for saving the tokenized data.

* data/tokenized
* data/cache

## Appearance features

* TGIF_RESNET_pool5.hdf5

### Other adjustments

*TODO: PATH:* There are paths to change in all 3 of dataset.py, main_frame.py and split_csv.py

*TRAIN/VAL SPLIT:* We use split_csv.py to perform a train/val split prior to training.

*NEW DROPOUT:* In order to reduce overfitting, we added new dropout layers to FrameQA_model.py, language_model.py and model_utils.py. These changes are marked with the comment ```NEW DROPOUT```

*Parameters:* We set lr=3e-3 and weight_decay=5e-6. For areas of the codebase marked with ```NEW DROPOUT```, we used a dropout probability of 0.15. We found these parameters best reduced overfitting.
