import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

dataset = pd.read_csv("preprocessedDatasets/finalPreprocessedDataset.csv", index_col=0)

# This graph is generating the number of features per column
# Loading variables for creating the graph
target_columns = [
    "orientation=no_orientation",
    "orientation=heterosexual",
    "orientation=homosexual",
    "orientation=bisexual",
]

distribution = dataset[target_columns].sum()

# Creating the graph

plt.figure(figsize=(10, 6))
distribution.plot(kind="bar")
plt.title("Distribution of Targets")
plt.xlabel("Class")
plt.ylabel("Sample Numbers")
plt.show()


# This is to visualise the most common words
feature_column = 'comment'

# Obtain feature column and the numbers of the msot frequent words
X = dataset[feature_column]
vectoriser = CountVectorizer(stop_words='english', max_features=20)
x_counts = vectoriser.fit_transform(X)
names = vectoriser.get_feature_names_out()
word_counts = X.count.toarray().sum(axis=0)

# graphing of the number of words graph
plt.figure(figsize=(10, 6))
plt.barh(range(len(word_counts)), word_counts, align="center")
plt.yticks(range(len(word_counts)), names)
plt.xlabel("Frequency")
plt.title("Top 20 Most Frequent Words in Comments")
plt.show()
