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

## Validation on additional data
An example of classification of an external dataset, in this case a selection of genomes downloaded from the NCBI GeneBank repository, using the model trained in the ‘Training and Testing the full HC model ’ has also been provided. It should take approx ~1.5Gb RAM and 20 sec to complete. It can be run using:
```
ipython RunValidationData.ipynb
```
Outputs will be produced in the ‘model_outputs’ directory of the GitHub repo and have the prefix “validation - “. These have already been generated for users that would simply like to view the outputs. The same range of outputs have been generated as detailed in the ‘Training on example data’ section above.
