import numpy as np
import cPickle as pickle
import re
import sys
import nltk
#nltk.download()
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def open_dataset():
    with open('docs.pkl', 'rb') as infile:
        train = pickle.load(infile)
    with open('tune.pkl', 'rb') as infile:
        tune = pickle.load(infile)
    try:
        with open('test.pkl', 'rb') as infile:
            test = pickle.load(infile)
    except IOError:
        raise IOError("Test set not live yet!")
    return train, tune, test


def pre_process_docs(doc):
	#USe regular expressions to do a find and replace
	letters_only = re.sub("[^a-zA-Z ]", "",doc)
	lower_case = letters_only.lower()        # Convert to lower case
	words = lower_case.split()               # Split into words
	sw = stopwords.words("english")
	words = [w for w in words if not w in stopwords.words("english")]
	return " ".join(words)

def loop_docs():
	data=open_dataset()
	cleaned_docs = [pre_process_docs(doc) for doc in data]
	return cleaned_docs

def vectorize():
	vectorizer = CountVectorizer(analyzer = "word",   \
		                           tokenizer = None,    \
		                           preprocessor = None, \
		                           stop_words = None,   \
		                           max_features = 5000) 
	Documents = loop_docs()
	train_data_features = vectorizer.fit_transform(Documents)
	train_data_features = train_data_features.toarray()
	vocab = vectorizer.get_feature_names()
	dist = np.sum(train_data_features, axis=0)
	bow = vectorizer.fit_transform(data)
  	return bow, vectorizer.get_feature_names()

def bow_from_docs():
    """Create bag of words from train and test documents"""
    train, tune, test = open_dataset()
    # Combine train, tune and test set while prepping docs
    size_train, size_tune, size_test = len(train), len(tune), len(test)
    dataset = train + tune + test
    dataset, idx2words = vectorize() 
    # Resplit train and test set ready for training
    train = dataset[:size_train]
    tune = dataset[size_train:(size_train + size_tune)]
    test = dataset[(size_train + size_test):]
    return train, tune, test, idx2words

def tune_lda(train, test):
    """
    Find stepsize from a prespecified set that minimizes tune set perplexity 
    (normally we might tune more constants but this serves as a demonstration)
    """
    decays = [0.7, 0.5, 0.1, 0.05, 0.01]
    sys.stdout.write("Tuning stepsizes (of {0}): ".format(len(decays)))
    scores = np.zeros(len(decays))
    for i, decay in enumerate(decays):
        sys.stdout.write("{0} ".format(i + 1))
        sys.stdout.flush()
        m = LDA(learning_method = 'batch')
        m.fit(train)
        scores[i] = m.perplexity(test)
    sys.stdout.write("\n")
    # Find best stepsize and return
    return decays[np.argmin(np.array(scores))]

def report_results(topic_words, idx2words):
    """Get 10 most popular word indexes associated with each topic discovered"""
    n_topics = topic_words.shape[0]
    for i in xrange(n_topics):
        idxs_curr = topic_words[i,:].argsort()[-10:][::-1]
        print "Topic {0}: {1}\n".format(i + 1, " ".join([idx2words[i] for i in idxs_curr]))

def fit_lda():
    train, tune, test, idx2words = bow_from_docs()
    # Tune using perplexity of held out 'tune' set
    stepsize = tune_lda(train, tune)
    # Run again with best stepsize for longer
    print "Training full model..."
    m = LDA(learning_method = 'batch', max_iter = 100)
    m.fit(train)
    # Return fitted model and report perplexity on test set
    print "Perplexity of final model: {0}\n".format(m.perplexity(test))
    report_results(m.components_, idx2words)

fit_lda()







