# Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

# Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# import string tokenization function from data manipulation file
from stringTokenizationFunc import stringTokenize

"""
To Do:
1. Fine tune the model by choosing between the models
"""


dataset = pd.read_csv("preprocessedDatasets/finalPreprocessedDataset.csv", index_col=0)

target_columns = [
    "orientation=no_orientation",
    "orientation=heterosexual",
    "orientation=homosexual",
    "orientation=bisexual",
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


# Evaluation by cross validation

cross_val_score = cross_val_score(classifier, X_tfidf, y, cv=5, scoring="accuracy")
print("CVal:", cross_val_score)
print("Mean CVal score:", cross_val_score.mean())

# Evaluation by grid search
param = {
    "estimator__C": [0.1, 1, 10, 100],
    "estimator__solver": ["liblinear", "saga"],
}
search_grid = GridSearchCV(
    OneVsRestClassifier(LogisticRegression(max_iter=1000)),
    param,
    cv=5,
    scoring="accuracy",
)
search_grid.fit(X_tfidf, y)
print("Best parameters:", search_grid.best_params_)
print("Best cv score:", search_grid.best_score_)

# the grid search takes the longest 😴
# Evaluation takes a while to run btw note to ted and baron
# ohohoh jaron ilysm for the pre processing u did, its so beautiful

# Make predictions!!!!!
y_pred = classifier.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    "Classification Report:\n", classification_report(y_test, y_pred, zero_division=0)
)

new_comment = "I hate faggots"
# new_comment = input("Enter the comment: ")
# Tokenising the new comment so better
new_comment_toke = stringTokenize(new_comment)
new_comment_not_toke = ""
for word in new_comment_toke:
    new_comment_not_toke += word + " "
comment_toke = vectorizer.transform([new_comment_not_toke[:-1]])
predictions = classifier.predict(comment_toke)

# return values
print(predictions)
if predictions[0][0] == 1:
    print("The message is not offensive to those of the LGBTQ community")
if predictions[0][1] == 1:
    print("The message is offensive to heterosexuals")
if predictions[0][2] == 1:
    print(
        "The message is offensive to those of the LGBTQ community especially to the homosexuals"
    )
if predictions[0][3] == 1:
    print(
        "The message is offensive to those of the LGBTQ community especially the bisexuals"
    )
