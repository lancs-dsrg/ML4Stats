import wikipedia as wiki
import sys
import cPickle as pickle

def scrape_wiki(n = 100):
    """Scrape n pages from wikipedia, save as list of plain text"""
    for i in xrange(n):
        # print progress
        sys.stdout.write("{0} ".format(i))
        sys.stdout.flush()
        docs = []
        docs.append(grab_page())
    sys.stdout.write("\n")
    with open("docs.pkl", 'wb') as out:
        pickle.dump(docs, out)

def grab_page(retries = 5):
    """
    Get text content from wikipedia page using wikipedia library

    Handle page errors (e.g. if page does not exist) by retrying with different page up to 5 times.
    """
    name = wiki.random()
    try:
        page = wiki.page(name).content
    # If error thrown, retry and reduce retry counter
    except (wiki.exceptions.DisambiguationError, wiki.exceptions.PageError):
        if retries > 1:
            grab_page(retries - 1)
        else:
            raise ValueError("Reached max number of retries")

if __name__ == '__main__':
    scrape_wiki(10)
