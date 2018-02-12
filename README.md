# ML for Statisticians

Brief intro to some less familiar ML concepts for statisticians. Slides for the talk in `ML4Stats.pdf`. This talk will be followed by a practical session next week.

## Before LDA Python session

- Install python on your machine (preferably 2)
- Install the following python packages: `wikipedia`, `nltk`, `sklearn`
- Clone this repo and run `python scrape_wiki.py` which will download 1000 wikipedia documents onto your machine. If it doesn't work let me know. This will take a while as it's quite a slow way of doing things so leave some time! Will bring a spare dataset to the session just in case someone doesn't get a chance to download.

## What we'll be doing

- Starting point is `open_dataset.py` which demos how to open a python 'pickle' file
- We'll work through the sections *Data Cleaning* and *Creating Features* from [this Kaggle tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial#part-1-for-beginners-bag-of-words). This will allow us to create the bag of words dataset.
- Next we'll use the `sklearn` package [LDA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) to perform LDA on our datasets. We'll use our dictionaries to look at our resulting topics.
- The one who gets the best log predictive score (perplexity) wins :).
