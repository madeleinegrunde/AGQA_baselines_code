# Code for AGQA-Decomp evaluation

This folder contains code to evaluate model performance for the Accuracy, CA, RWR and IC metrics on the AGQA-Decomp benchmark. For detail on how these metrics are defined, please consult Sections 4 and 7.2 of our paper. We assume that models have been trained and evaluated with the .csv files found [here](https://agqa-decomp.cs.washington.edu), as was described in the main folder of this repository.

## Evaluation instructions
The scripts assume that:
* The folder containing question decompositions for the balanced AGQA test set is downloaded from [here](https://agqa-decomp.cs.washington.edu) and placed to this folder. The folder should be named ```balanced_test_hierarchies```
* Model predictions are placed in a folder named ```model_preds```. Files in that folder should be named according to the template ```[MODEL NAME]_preds.json```
* For each script, the contents of the lists marked with the comment ```TODO: MODEL NAME``` should be changed to reflect the names of the models being evaluated.

Afterwards, run the scripts in the following order to perform evaluation:
* preprocess.py
* compute_acc_ca_rwr.py
* compute_ic.py

## Scripts
### compute_acc_ca_rwr.py
This script computes model performances on the Accuracy, CA and RWR metrics, as well as the RWR-n and Delta metrics. Model performances are saved as .csv files in the analysis_results folder. Specifically,
* Accuracy performances are saved in the ```analysis_results/per_gt_acc``` folder under the file ```normalized_acc_[MODEL NAME].csv``` Sample sizes for question type/ground-truth answer pairs can be found on ```raw_counts_[MODEL NAME].csv```
* CA and RWR performances are saved in the ```analysis_results/ca``` and ```analysis_results/rwr``` folders respectively. The file ```ca_[MODEL NAME]_by_composition.csv``` will contain CA scores when conditioning on composition rules while ```ca_[MODEL NAME]_by_parent.csv``` will contain CA scores when conditioning on the parent question type. The naming convention is the same for RWR.
* RWR-n performances are also saved in the ```analysis_results/rwr``` folder. The file starting with ```rwr_n_values``` will contain RWR-n scores, while the file starting with ```rwr_n_counts``` will contain the sample sizes associated with each RWR-n value.

### compute_ic.py
This script computes model performances on the IC metric. Performances are saved in the ```analysis_results/ic``` folder. Files starting with ```per_composition``` will contain IC scores when conditioning on composition rules while those starting with ```per_parent``` will contain IC scores when conditioning on parent question types. Model performances on individual logical consistency rules can be found in files starting with  ```per_rule```

### ic_acc_correlation.py
This script computes the correlation between model accuracy and IC scores across a single DAG, as described in Section 5.6 of the paper.

### generate_analysis_splits.py
After running the primary evaluation scripts, this script can help perform qualitative analysis. For each model specified, it will collect copies of the compositions involved in the computation of the CA, RWR and RWR-n metrics. The ca_rwr_qualitative_analysis.ipynb notebook can then be used to visualize these compositions.
