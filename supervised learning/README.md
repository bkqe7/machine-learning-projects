# Finding Donors for Charity

The goal of this project is to build a machine learning model that accurately predicts whether an individual makes more than $50,000 using data collected from the 1994 U.S. Census. The supervised learning model is chosen among Logistic Regression, Random Forest and Adaboost Classifier by comparing the F-beta score and tuning the model parameters using cross validation. The final optimized model is presented with accuracy 0.8443 and F-score 0.6856 on the test set.

## Installation

The packages used are:

- Numpy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Time

## Files and How to Use Them

- The finding_donors.ipynb contains EDA, feature engineering, model selection, model tuning and extracting feature importance. 
- The visuals.py provides visulization modual for feature engineering.
- The census.csv is the 1994 U.S. Census data.

## Liscensing, Authors and Acknowledgements

The dataset for this project originates from the UCI Machine Learning Repository. It is donated by Ron Kohavi and Barry Becker.
