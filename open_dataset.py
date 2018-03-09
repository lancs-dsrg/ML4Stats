import cPickle as pickle

def open_dataset():
    # Quick demo to show you how to load the dataset you created :)
    with open('docs.pkl', 'rb') as infile:
        data = pickle.load(infile)
    return data

if __name__ == '__main__':
    data = open_dataset()
    # Output dataset
    print data
