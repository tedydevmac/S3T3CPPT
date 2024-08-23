# Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

"""
To Do:
1. Make predictions with the model
"""

dataset = pd.read_csv("./finalPreprocessedDataset.csv", index_col=0)

target_columns = [
    "orientation=no_orientation",
    "orientation=heterosexual",
    "orientation=homosexual",
]

feature_column = "comment"

# featores
X = dataset[feature_column]
y = dataset[target_columns]

# make into numerical data
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split the data into the sets
x_train, x_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Training the weak classifier
classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
classifier.fit(x_train, y_train)

# Evaluate model :)))
y_pred = classifier.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    "Classification Report:\n", classification_report(y_test, y_pred, zero_division=0)
)


# Make predictions!!!!!
# new_comment = input("Enter the comment: ")
new_comment = "i hate faggot"

comment_toke = vectorizer.transform([new_comment])
predictions = classifier.predict(comment_toke)
print(predictions)
