# This file gives an example solution for you to compare to.
import cPickle as pickle
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer


def create_bow(docs):
    vectorizer = CountVectorizer(max_features = 1000) 
    train = vectorizer.fit_transform(docs)


def prep_docs(docs):
    """Prepare documents to be converted to bag of words"""
    # Convert stopwords to set for speed
    stopws = set(stopwords.words("english"))
    data = [clean_doc(doc, stopws) for doc in docs]
    return data


def clean_doc(doc, stopws):
    """Strip punctuation etc. from a single document"""
    doc = doc.lower()
    # Strip punctuation
    doc = re.sub(r'-', ' ', doc)
    doc = re.sub(r'[^a-z ]', '', doc)
    doc = re.sub(r' +', ' ', doc)
    # Split document to words and remove stopwords, rejoin
    return " ".join([word for word in doc.split() if word not in stopws])


def open_dataset():
    with open('docs.pkl', 'rb') as infile:
        data = pickle.load(infile)
    return data


if __name__ == '__main__':
    data = open_dataset()
    data = prep_docs(data)
    create_bow(data)
