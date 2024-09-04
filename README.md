# Cyberbullying Detection Project

This project focuses on detecting and classifying cyberbullying in text, particularly against the LGBTQ community.

## Stage 1 - Gather Data

No scripts

## Stage 2 - Data Pre-Processing

### Dataset 1

[Dataset 1 Preprocessing Code](./datasetOnePreprocessing.py) -  preprocessing the dataset from the [Cyberbullying Data for Multi-Label Classification](https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification?select=final_hateXplain.csv).

### Dataset 2

[Dataset 2 Preprocessing Code](./datasetTwoPreprocessing.py) - preprocessing the dataset from the [Anti LGBTQ Cyberbullying Texts](https://www.kaggle.com/datasets/kw5454331/anti-lgbt-cyberbullying-texts) dataset.

### After Concatenating Datasets

[Joint Dataset Preprocessing Code](./jointDatasetPreprocessing.py) - Concantenates the 2 datasets

### Data Visualisation

[Data Visualisation Code](./visualise.py) - Creates the visualisation

Run `datasetOnePreprocessing.py` and `datasetTwoPreprocessing.py` before running `jointDatasetPreprocessing.py`.

## Stage 3, 4 & 5 - Model Selection, Training, Evaluation, and Prediction

### Logistic Regression

[Logistic Regression Code](./logisticRegression.py) - The logistic regression model

### K-Nearest Neighbours

[K-Nearest Neighbours Code](./knn.py) - The k-nearest neighbours model

---

## Additional Files

- Function for tokenising: Used in  like `datasetOnePreprocessing.py`, `datasetTwoPreprocessing.py`, and `logisticRegression.py`.
  - [stringTokenizationFunc.py](./stringTokenizationFunc.py)
