import numpy as np
import pandas as pd

datasetOnePreprocessed = pd.read_csv("preprocessedDatasets/datasetOnePreprocessed.csv")
datasetTwoPreprocessed = pd.read_csv("preprocessedDatasets/datasetTwoPreprocessed.csv")

jointDataset = pd.concat([datasetOnePreprocessed, datasetTwoPreprocessed])
jointDataset = jointDataset.drop(['Unnamed: 0'],axis=1)



print('No. of true values per column:\n',jointDataset.apply(lambda x: (x == True).sum()))
print('No. of false values per column:\n',jointDataset.apply(lambda x: (x == False).sum()))
print()
# asexual has only 4 values of true, also dataset two doesn't have asexual, meaning that there are many missing values in asexual

jointDataset = jointDataset.drop(['orientation=asexual'],axis=1)

print('No. of true values per column before deleting asexual:\n',jointDataset.apply(lambda x: (x == True).sum()))
print('No. of false values per column after deleting asexual:\n',jointDataset.apply(lambda x: (x == False).sum()))
print()
# all targets in dataset have a decent number of true and false values

print('First 5 values of joint preprocessed dataset transposed:\n',jointDataset.head().T)
jointDataset.to_csv("preprocessedDatasets/finalPreprocessedDataset.csv")