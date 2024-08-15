# Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split


"""
To Do:
1. Load x and y 
2. define train test split
3. Linear regression 
4. Evaluate model
"""

# csv reading
dataset = pd.read_csv("finalPreprocessedDataset.csv", index_col=0)

# features
target_columns = [
    "orientation=no_orientation",
    "orientation=asexual",
    "orientation=heterosexual",
    "orientation=homosexual",
    "orientation=bisexual",
]

x = dataset.drop(target_columns, axis=1)
y = dataset[target_columns]
