import numpy as np
import pandas as pd

from stringTokenizationFunc import stringTokenize

dataset = pd.read_csv("rawDatasets/rawDatasetOne.csv")
print("Dataset (First 5 Values Transposed):\n", dataset.head().T)  # look at dataset
print()

# Check for Datatypes
print("Column Datatypes:\n", dataset.dtypes)  # Datatypes are all objects
print()

# Converting datatypes to correct datatypes
dataset["comment"] = dataset["comment"].astype("string")
dataset["label"] = dataset["label"].astype("string")
dataset["Race"] = dataset["Race"].astype("string")
dataset["Religion"] = dataset["Religion"].astype("string")
dataset["Gender"] = dataset["Gender"].astype("string")
dataset["Sexual Orientation"] = dataset["Sexual Orientation"].astype("string")
dataset["Miscellaneous"] = dataset["Miscellaneous"].astype("string")

print("Processed Column Datatypes:\n", dataset.dtypes)  # Datatypes are correct now
print()

# Checking Column Names
print(
    "Dataset Column Names:\n", dataset.columns
)  # Column capitalisation is inconsistent, one of the names is two words
print()

dataset = dataset.rename(str.lower, axis="columns")
dataset = dataset.rename(columns={"sexual orientation": "orientation"})

print(
    "Processed Dataset Column Names:\n", dataset.columns
)  # All columns are uncapitalised and 1 word
print()

# Checking for missing values
print(
    "No. of missing values per column in dataset:\n", dataset.isnull().sum()
)  # miscellaneous column has many missing values
print()

dataset.drop(columns=["miscellaneous"], inplace=True)  # Too few values to be a target

print(
    "No. of missing values per column in dataset after dropping miscellaneous column:\n",
    dataset.isnull().sum(),
)  # No more missing values
print()

# Checking for bad data
# Replacing common terms used to denote missing value
for column_name in dataset.columns:
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x == "?" else x
    )
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x == "-" else x
    )
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x.lower() == "n.a." else x
    )
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x.lower() == "n.a" else x
    )
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x.lower() == "na" else x
    )
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x.lower() == "na." else x
    )
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x == "0" else x
    )
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x.lower() == "nil" else x
    )
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x.lower() == "none" else x
    )
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x.lower() == "null" else x
    )
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x.lower() == "nan" else x
    )

print(
    "No. of missing values per column in dataset after replacement of common terms used to denote missing value:\n",
    dataset.isnull().sum(),
)  # Dataset has no missing values
print()

# Our only target sexual orientation, so we drop the rest of the targets
dataset.drop(columns=["label", "race", "gender", "religion"], inplace=True)

# Converting all values in dataset to lowercase
for column_name in dataset.columns:
    dataset[column_name] = dataset[column_name].str.lower()

# One-Hot Encoding (Targets)
uniqueTargetValues = {}
encodedDataset = pd.DataFrame(dataset["comment"])
for (
    column_name
) in dataset.columns:  # Finding all the unique column values in the dataset
    if column_name != "comment":
        uniqueTargetValues[column_name] = set(dataset.loc[:, column_name].tolist())
print("All unique  values in dataset:\n", uniqueTargetValues)
print()

for column_name in dataset.columns:
    dataset[column_name] = dataset[column_name].apply(
        lambda x: np.nan if x == "?" else x
    )

for column_name in uniqueTargetValues:  # Creating a new column in the dataset
    for itm in uniqueTargetValues[column_name]:
        tempNewCol = pd.DataFrame(
            {(column_name + "=" + itm): (list(dataset[column_name] == itm))}
        )
        encodedDataset = pd.concat([encodedDataset, tempNewCol], axis=1)
        tempNewCol = None
print("One Hot Encoded Dataset (First 5 Values Transposed):\n", encodedDataset.head().T)
print()

# string Tokenization, remove punctuation, stopwords, emojis, doing word stemming and replacing common text abbreviations

tokenizedDataset = encodedDataset["comment"].apply(stringTokenize)
tokenizedDataset = pd.concat(
    [tokenizedDataset, encodedDataset.loc[:, encodedDataset.columns != "comment"]],
    axis=1,
)

print(
    "First 5 values of final preprocessed tokenized one-hot encoded dataset transposed:\n",
    tokenizedDataset.head().T,
)
tokenizedDataset.to_csv("preprocessedDatasets/datasetOnePreprocessed.csv")
