# import final dataset into main file

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('S3T3CPPT/final_hateXplain.csv')
print('Dataset Head Transposed:\n',dataset.head().T) # look at dataset
print()

'''
1.Check for Data Types
2.Check Columns Names
3.Check for Missing Values
4.Check for Bad Data 
5.Imputation of Null values
6.Check for distribution type
7.Scaling the data 
8.Checks for outliers 
9.Check for data Imbalance 
10.Perform necessary transformations
11.Perform feature Engineering 
12.Binning Continuous data 
13.Feature selection
'''

# Check for Datatypes
print('Column Datatypes:\n',dataset.dtypes) # Datatypes are all objects
print()

# Converting datatypes to correct datatypes
dataset['comment'] = dataset['comment'].astype('string')
dataset['label'] = dataset['label'].astype('string')
dataset['Race'] = dataset['Race'].astype('string')
dataset['Religion'] = dataset['Religion'].astype('string')
dataset['Gender'] = dataset['Gender'].astype('string')
dataset['Sexual Orientation'] = dataset['Sexual Orientation'].astype('string')
dataset['Miscellaneous'] = dataset['Miscellaneous'].astype('string')

print('Processed Column Datatypes:\n',dataset.dtypes) # Datatypes are correct now
print()

# Checking Column Names
print('Dataset Column Names:\n',dataset.columns) # Column capitalisation is inconsistent, one of the names is two words
print()

dataset = dataset.rename(str.lower, axis='columns')
dataset = dataset.rename(columns={'sexual orientation':'orientation'})

print('Processed Dataset Column Names:\n',dataset.columns) # All columns are uncapitalised and 1 word
print()

# Checking for missing values
print('No. of missing values per column in dataset:\n',dataset.isnull().sum()) # miscellaneous column has many missing values
print()

dataset.drop(columns=['miscellaneous'],inplace=True) # Too few values to be a target

print('No. of missing values per column in dataset after dropping miscellaneous column:\n',dataset.isnull().sum()) # No more missing values
print()

# Checking for bad data
# Replacing common terms used to denote missing value
for column_name in dataset.columns:
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x == '?' else x)
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x == '-' else x)
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x.lower() == 'n.a.' else x)
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x.lower() == 'n.a' else x)
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x.lower() == 'na' else x)
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x.lower() == 'na.' else x)
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x == '0' else x)
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x.lower() == 'nil' else x)
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x.lower() == 'none' else x)
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x.lower() == 'null' else x)
    dataset[column_name] = dataset[column_name].apply(lambda x: np.nan if x.lower() == 'nan' else x)

print('No. of missing values per column in dataset after replacement of common terms used to denote missing value:\n',dataset.isnull().sum()) # Dataset has no missing values
print()

# One-Hot Encoding
uniqueColumnValues = {}
oldColumns = dataset.columns
for column_name in dataset.columns:
    print(column_name)
    if column_name != 'comment':
        uniqueColumnValues[column_name] = set(dataset.loc[:, column_name].tolist())
print(uniqueColumnValues)
for column_name in uniqueColumnValues:
    for itm in uniqueColumnValues[column_name]:
        tempNewCol = pd.DataFrame({(column_name+'='+itm):(list(dataset[column_name] == itm))})
        dataset = pd.concat([dataset,tempNewCol])
dataset.drop(columns=dataset.columns)
dataset.to_csv('output.csv')
