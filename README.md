No stage: stringTokenizationFunc.py
Used in datasetOnePreprocessing.py, datasetTwoPreprocessing.py, logisticRegression.py

Stage 1 - Gather Data
In report, not in code

Stage 2 - Data Pre-Processing
Dataset 1: [datasetOnePreprocessing.py]('./datasetOnePreprocessing.py) - dataset preprocessing for the dataset on [cyberbullying data for multi label classification](https://www.kaggle.com/datasets/sayankr007/cyber-bullying-data-for-multi-label-classification?select=final_hateXplain.csv)
Dataset 2: [datasetTwoPreprocessing.py]('./datasetTwoPreprocessing.py) - dataset preprocessing for the dataset on [anti lgbtq cyberbullying texts](https://www.kaggle.com/datasets/kw5454331/anti-lgbt-cyberbullying-texts)
After concatenating datasets: [jointDatasetPreprocessing.py]('./jointDatasetPreprocessing.py')
Data visualisation: [visualise.py]('./visualise.py)
Run datasetOnePreprocessing.py and datasetTwoPreprocessing.py before jointDatasetPreprocessing.py

Stage 3 - Choose and Train Model, Stage 4 - Evaluate and Tune Model and Stage 5 - Make Predictions
Logistic Regression model: [logisticRegression.py]('./logisticRegression.py')
K-Nearest-Neighbours: [knn.py]('./knn.py)
Stage 3, 4 and 5 are all in the same file (indicated by comments)
