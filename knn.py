# Dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
import numpy as np

# Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# import string tokenization function from data manipulation file

from stringTokenizationFunc import stringTokenize

# Stage 3 - Choose and Train Model

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
knn = KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree")
knn.fit(x_train, y_train)

# ohohoh jaron ilysm for the pre processing u did, its so beautiful

y_pred = knn.predict(x_test)

# Stage 4 - Evaluate and Tune Model

# Evaluation by cross validation
cross_val_score = cross_val_score(knn, X_tfidf, y, cv=5, scoring="accuracy")
print("CVal:", cross_val_score)
print("Mean CVal score (use this):", cross_val_score.mean())

# Evaluation by Confusion Matrix
conf_matrix = confusion_matrix(y_test.values.argmax(axis=1), y_pred.argmax(axis=1))
print("Confusion Matrix:\n", conf_matrix)
conf_matrix_np = np.array(conf_matrix)
# Calculate the sum of TP and TN 
TP = np.diag(conf_matrix)
FP = np.sum(conf_matrix, axis=0) - TP
FN = np.sum(conf_matrix, axis=1) - TP
TN = np.sum(conf_matrix) - (FP + FN + TP)
# Calculate the total number of samples
total_samples = np.sum(conf_matrix)
# Calculate the accuracy
accuracy = (np.sum(TP) + np.sum(TN))/ (np.sum(TP) + np.sum(TN) + np.sum(FP) + np.sum(FN))
print("Confusion Matirx %: ", accuracy)

# Stage 5 - Make Predictions


# predictions on user's input

new_comment = ""
while new_comment.lower() != 'quit':
    new_comment = input('Enter the comment: (Enter "quit" to exit)')
    # string Tokenization, remove punctuation, stopwords, emojis, doing word stemming and replacing common text abbreviations
    new_comment_toke = stringTokenize(new_comment)
    new_comment_not_toke = ""
    # merging tokenised string into one sentence for vectorizer.transform to work
    for word in new_comment_toke:
        new_comment_not_toke += word + " "
    comment_toke = vectorizer.transform([new_comment_not_toke[:-1]])
    predictions = knn.predict(comment_toke)

    # return values to see who the message is offensive to or not offensive
    print(predictions)
    for i in predictions: 
        # if sum(i) > 1:
        #     print("The message is not offensive to those of the LGBTQ community") # e,g. "homophobic"
        if sum(i) == 0:
            print("The message is offensive to those part of the TQ+ community of the LGBTQ+") # e.g. "trans"
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