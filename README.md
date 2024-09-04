# Cyberbullying Detection Project

This project focuses on detecting and classifying cyberbullying in text, particularly against the LGBTQ community.

## Stage 1 - Gather Data

No code

## Stage 2 - Data Pre-Processing

### Dataset 1

[datasetOnePreprocessing.py](./datasetOnePreprocessing.py) -  preprocessing the dataset from the [Cyberbullying Data for Multi-Label Classification](https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification?select=final_hateXplain.csv).

### Dataset 2

[datasetTwoPreprocessing.py](./datasetTwoPreprocessing.py) - preprocessing the dataset from the [Anti LGBTQ Cyberbullying Texts](https://www.kaggle.com/datasets/kw5454331/anti-lgbt-cyberbullying-texts) dataset.

### After Concatenating Datasets

[jointDatasetPreprocessing.py](./jointDatasetPreprocessing.py) - Concantenates the 2 datasets

### Data Visualisation

[visualise.py](./visualise.py) - Creates the visualisation

Run `datasetOnePreprocessing.py` and `datasetTwoPreprocessing.py` before running `jointDatasetPreprocessing.py`.

## Stage 3, 4 & 5 - Model Selection, Training, Evaluation, and Prediction

### Logistic Regression

[logisticRegression.py](./logisticRegression.py) - The logistic regression model

### K-Nearest Neighbours

[knn.py](./knn.py) - The k-nearest neighbours model

---

## Additional Files

- Function for tokenising: Used in  like `datasetOnePreprocessing.py`, `datasetTwoPreprocessing.py`, and `logisticRegression.py`.
  - [stringTokenizationFunc.py](./stringTokenizationFunc.py)
