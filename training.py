# Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

"""
To Do:
1. Load x and y 
2. define train test split
3. Train a classifier
4. Evaluate model
"""

# csv reading
dataset = pd.read_csv("./finalPreprocessedDataset.csv", index_col=0)

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

# Features and target
X = dataset[feature_column]
y = dataset[target_columns]

# Convert text data to numerical data using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train a classifier using OneVsRestClassifier
classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
classifier.fit(x_train, y_train)

# Evaluate the model
y_pred = classifier.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))