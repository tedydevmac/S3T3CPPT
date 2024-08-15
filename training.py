# Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

"""
To Do:
1. Load x and y 
2. define train test split
3. Train a classifier
4. Evaluate model
"""

# csv reading
dataset = pd.read_csv("finalPreprocessedDataset.csv", index_col=0)

# Define target columns (orientation columns)
target_columns = [
    "orientation=no_orientation",
    "orientation=asexual",
    "orientation=heterosexual",
    "orientation=homosexual",
    "orientation=bisexual",
]

# Define feature column (comments column)
feature_column = "comment"

# Features
x = dataset[[feature_column]]

# Targets
y = dataset[target_columns]

# Training and Testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)  # 75% training, 25% testing

classifier = LogisticRegression(max_iter=1000, multi_class="ovr")

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
