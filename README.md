# Housing Price Project
Version 1.0

Summary of Project:

This project will explore how feature selection and regularization (L1 & L2) methods influence the prediction scores of regression models.

The problem in the Kaggle housing <a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data">data</a> is to predict what price a house sold for using the available features.  This could be useful information for a potential home buyer.  

The focus for this workbook however will be to use the dataset to explore how various feature selection and regularization (L1 & L2) methods influence predicted housing prices when applying regression models.  


### Table of Contents

1. [Installation](#installation)
2. [Instructions](#instructions)
3. [Project Motivation](#motivation)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

See environment.yml for full list of packages used.  The package manager used was conda and coded in Python 3.12.2.  The main packages used were pandas, sci-kit learn, and streamlit (for the app).

You can use the environment.yml to create an exact anaconda virtual environment with the command `conda env create -f environment.yml`

## Instructions <a name="instructions"></a>
In order to run the app follow the below instructions.

1. Download the files listed in the File Descriptions section below to a folder on your machine.
2. In a terminal, create a conda environment using the yaml file with `conda env create -f environment.yml`.
3. Activate the environment with `conda activate housing`.
4. Run the clean data script with ` python clean_data.py`.  This step will save a couple of pickle files with the necessary X and y matrix (X.pkl & y.pkl) along with 3 joblib files containing the sklearn pipelines (features_only.joblib, numerical_features_PCA.joblib, combined_features.joblib).
5. Activate the app using the command `streamlit run app.py`.

## Project Motivation<a name="motivation"></a>

The motivation for the project originated with my interest in how models can be fine-tuned to improve their performance.  I wanted to hone in on feature selection and regularization particularly so that their impact could be illustrated.  It is also a good learning opportunity.

## File Descriptions <a name="files"></a>

* housing.ipynb - Main notebook file containing the main analysis and conclusions.
* train.csv - Training data used for the bulk of the analysis.
* environment.yml - List of packages used.
* clean_data.py - File which transforms the training data and creates several files necessary for the app.
* app.py - Code for the Streamlit app.
* functions.py - Helper functions for the Streamlit app.

## Results<a name="results"></a>

As can be seen in the notebook visualizations, using feature selection or regularization can greatly improve the prediction power of regression models.  This can be attributed to these models being more robust to new unseen data as compared to the models without feature selection or regularization.  When feature selection is not performed the model tends to over-fit and this can lead to poor results in the general case. 

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Please feel free to use the code here as you like.  I acknowledge Kaggle for their dataset.
