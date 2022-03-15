# Hierarchical Machine Learning (ML)
This repository contains the required python scripts and associated data to train and test a Hierarchical Machine Learning (ML) model using most classifiers and resamplers supported by the python ski-learn and associated packages. The majority of scripts have been provided as jupyter notebook files (.ipynb) to enhance users ability to break the scripts down into manageable and understandable sections.

## Associated Publication
The methodology underlying the package has been detailed in ***. 

## Dependencies 
The dependencies for all scripts presented herein can be installed most efficiently using conda and the specification file provided in the repo:

```
conda create --name Hierarchical_ML --file spec-file.txt
conda activate  Hierarchical_ML
```
## The HC package

The scripts for running a Hierarchical classifier (HC) have been provided in the ‘HC_package’ directory. Notes and annotation for individual functions have been provided in the HierarchicalClassification.py script. NOTE: this is not a fully scikit-learn integrated package, rather it is a collection of scripts to support a specific implementation of a Hierarchical classification algorithm as detailed in the associated publication. 

## inputs 
The required inputs for the model include:
1/ feature file - a binary (present/absent) file containing named features for machine learning (columns) per sample (rownames)
2/ labels file -  a file containing one label per row corresponding to the order of the samples in the features file.
3/ graph file – a directed acyclic graph generated using the networkx package.

Examples of these files have been provided in the repository. These are the inputs used to train the model presented in the associated publication. These example files were generated from unitig data created from filtered paired fastq files. They were subsequently processed into ‘patterns’ to reduce the size of the dataset. Patterns represent groups of features which have perfect correlation, i.e. they occur in identically sets of isolates. 

## Optimised model
A fully optimised HC model, presented in the main manuscript has been mae available in the ***'optimised_model'*** directory. The model parameters were optimised using a genetic algorithm as implemented in [TPOT](https://github.com/EpistasisLab/tpot). Additional details have been provided in the main manuscript. This directory contains a pickled file containing:

-***models*** - an optimised HC model. 
-***train_features*** - features (patterns) used in the final model. 
-***graph*** - the hierachical graph (DAG) used in the final model. 

The model and associated data can be read and loaded in python using:
```
# load model and associated data 
pkl = "./HierarchicalML/optimised_model/optimised_model_data.pkl"
with gzip.open(pkl, 'rb') as f:
    
    models = pickle.load(f)
    train_features = pickle.load(f)
    graph = pickle.load(f)
```

## Training on example data
An example data has been created by taking the first ~100 features from the main dataset. It should run quickly (~40 secs) on even limited hardware and take ~400 Mb RAM. It can be run in full using:

```
ipython TrainHCModel_example.ipynb
```
Outputs will be produced in the ‘example outputs’ directory of the GitHub repo. These will include:

**models.pkl** – pickled trained hierarchical classifier models and associated data.
**training_summary.tsv** – summary of hierarchical summary statistics (hP, hR, hF1) calculated per class  for the training dataset.
**test_summary.tsv** – summary of hierarchical summary statistics (hP, hR, hF1) calculated per class for the test dataset.
**per_class_summary.tsv** – summary of the non-hierarchical summary statistics (precision, recall, F1 score) calculated per class for the test dataset.
**per_node_summary.tsv** – summary of the non-hierarchical summary statistics(recall, precision, accuracy, micro, weighted) calculated per node for the test dataset.


## Training and Testing the full HC model 
To train the full model will take approx 10-2n mins on a desktop computer and require ~15Gb RAM. The model is trained using a train/test split of 0.75/0.25. It can be run in full using:

```
ipython TrainHCModel.ipynb
```
Outputs will be produced in the ‘model_outputs’ directory of the GitHub repo. These have already been generated for users that would simply like to run validation data (see below). The same range of outputs have been generated as detailed in the ‘Training on example data’ section above.


## Validation datasets 
A range of validation datasets have been used to test the efficacy of the model on external data. The 5 datasets used in the manuscript have been provided as raw pattern data and can be classified using jupyter/ipython notebook scripts. Validation data and scripts can be found in the ***'validation data'*** directory, with one directory per dataset. Each directory has the same structure containing:
- ***sample_list.txt***: a list of samples (SRR codes) to process.
- ***sample.location***: a tab seperated list of sample names and location (class) information.
- ***patterns.tab***: the patterns generated from unitigs used as features in te model
- ***classify_new_samples.ipynb***: jupyter notebook containig script to generate outputs

The classify_new_samples.ipynb scripts should take less than ~1.5Gb RAM and ~20 sec to complete. The validation data will be processed using the fully optimised model presented in the manuscript. The individual scripts can be run using:
```
ipython classify_new_samples.ipynb
```
Outputs will be produced in the appropriate validation dataset directory. ***Note:*** these have already been generated for users that would simply like to view the outputs. These outputs include: 

- ***output_hsummary.tab*** – overall summary of hierarchical summary statistics (hP, hR, hF1) for the validation dataset.
- ***output_per_class.tsv*** - summary of hierarchical summary statistics (precision, recall, F1 score) calculated per class for the validation dataset.
- ***output_nonhier_per_class.tsv*** - summary of non-hierarchical summary statistics (precision, recall, F1 score) calculated per class for the validation dataset.
- ***output_per_sample.tsv*** - output containing the true label and predited class per sample as well as the predicted probabilities of the model at each node in the hierarchy. Note: only nodes investigated by the model will be shown (i.e. if a sample is classified into Europe at the regional root node it will not have a predicted probability for South-east Asia, only for European subregions).  

**training_summary.tsv** – summary of hierarchical summary statistics (hP, hR, hF1) calculated per class  for the training dataset.
**test_summary.tsv** – summary of hierarchical summary statistics (hP, hR, hF1) calculated per class for the test dataset.
**per_class_summary.tsv** – summary of the non-hierarchical summary statistics (precision, recall, F1 score) calculated per class for the test dataset.
**per_node_summary.tsv** – summary of the non-hierarchical summary statistics(recall, precision, accuracy, micro, weighted) calculated per node for the test dataset.
