import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import ssl

# downloading nltk resources
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')

# string Tokenization, remove punctuation, stopwords, emojis, doing word stemming and replacing common text abbreviations
def stringTokenize(string):
    # removing the <user> tags at the start
    string = re.sub("<(\w+)>", "", string)

    # replacing common text abbreviations
    string = re.sub(" u ", " you ", string)
    string = re.sub(" ikr ", " i know right ", string)
    string = re.sub(" idk ", " i do not know ", string)
    string = re.sub(" lol ", " laugh out loud ", string)
    string = re.sub(" ik ", " i know ", string)

    # tokenize and remove punctuation
    puncInRegex = re.sub(r"[^\s\w]", "", string)
    removePuncTokenized = word_tokenize(puncInRegex.lower())

    # remove stopwords
    stopWords = stopwords.words("english")
    stopWordsRemoved = []
    for itm in removePuncTokenized:
        if itm not in stopWords:
            stopWordsRemoved.append(itm)

    # stemming
    stemmed = []
    ps = PorterStemmer()
    for w in stopWordsRemoved:
        stemmed.append(ps.stem(w))
    return stemmed

print(stringTokenize('hello world'))