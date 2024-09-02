import numpy as np
import pandas as pd

from stringTokenizationFunc import stringTokenize

dataset = pd.read_csv("rawDatasets/rawDatasetTwo.csv")
# deleted unnecessary/unrelated columns from raw dataset in excel
print("Dataset (First 5 Values Transposed):\n", dataset.head().T)  # look at dataset
print()

# Check for Datatypes
print("Column Datatypes:\n", dataset.dtypes)  # Datatypes are all correct, except "text"
print()

# Converting datatypes to correct datatypes
dataset["text"] = dataset["text"].astype("string")

print("Processed Column Datatypes:\n", dataset.dtypes)  # "text" datatype is correct now
print()

# Checking Column Names
print(
    "Dataset Column Names:\n", dataset.columns
)
print()

dataset = dataset.drop(['target_sexuality_other'],axis=1) # column name too ambiguous
dataset = dataset.rename(columns={"text": "comment","target_sexuality_straight":"orientation=heterosexual","target_sexuality_bisexual":"orientation=bisexual"}) # renaming according to preprocessed version of dataset one, heterosexual will be done later
dataset["orientation=no_orientation"] = ~dataset['target_sexuality']
dataset = dataset.drop(['target_sexuality'],axis=1)
dataset['orientation=homosexual'] = dataset['target_sexuality_gay'] | dataset['target_sexuality_lesbian'] # merging gay and lesbian into homosexual
dataset = dataset.drop(['target_sexuality_gay','target_sexuality_lesbian'],axis=1)

print(
    "Processed Dataset Column Names:\n", dataset.columns
)  # All columns are aligned with preprocessed version of dataset one now
print()

# Checking for missing values
print(
    "No. of missing values per column in dataset:\n", dataset.isnull().sum()
)  # no missing values
print()

# Checking for bad data
# Replacing common terms used to denote missing value (for comments only as everything else isn't string)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x == "?" else x
)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x == "-" else x
)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x.lower() == "n.a." else x
)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x.lower() == "n.a" else x
)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x.lower() == "na" else x
)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x.lower() == "na." else x
)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x == "0" else x
)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x.lower() == "nil" else x
)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x.lower() == "none" else x
)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x.lower() == "null" else x
)
dataset['comment'] = dataset['comment'].apply(
    lambda x: np.nan if x.lower() == "nan" else x
)

print(
    "No. of missing values per column in dataset after replacement of common terms used to denote missing value:\n",
    dataset.isnull().sum(),
)  # Dataset has no missing values
print()

# Converting all values in comment to lowercase
dataset['comment'] = dataset['comment'].str.lower()

# Targets are already one-hot encoded

# string Tokenization, remove punctuation, stopwords, emojis, doing word stemming and replacing common text abbreviations

tokenizedDataset = dataset["comment"].apply(stringTokenize)
tokenizedDataset = pd.concat(
    [tokenizedDataset, dataset.loc[:, dataset.columns != "comment"]],
    axis=1,
)

print(
    "First 5 values of final preprocessed tokenized one-hot encoded dataset transposed:\n",
    tokenizedDataset.head().T,
)
tokenizedDataset.to_csv("preprocessedDatasets/datasetTwoPreprocessed.csv")
