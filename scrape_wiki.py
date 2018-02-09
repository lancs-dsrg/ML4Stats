import wikipedia as wiki
import sys
import cPickle as pickle

def scrape_wiki(n = 10 ** 3):
    """Scrape n pages from wikipedia, save as list of plain text"""
    docs = []
    sys.stdout.write("Num docs of {0}: ".format(n))
    for i in xrange(n):
        # print progress
        sys.stdout.write("{0} ".format(i + 1))
        sys.stdout.flush()
        docs.append(grab_page())
    sys.stdout.write("\n")
    # Output to python's data file
    with open("docs.pkl", 'wb') as out:
        pickle.dump(docs, out)

def grab_page(retries = 5):
    """
    Get text content from wikipedia page using wikipedia library

    Handle page errors (e.g. if page does not exist) by retrying with different page up to 5 times.
    """
    # Get random wikipedia page name and try to download the content
    name = wiki.random()
    try:
        page = wiki.page(name).content
    # If error thrown, retry and reduce retry counter
    except (wiki.exceptions.DisambiguationError, wiki.exceptions.PageError):
        if retries > 1:
            page = grab_page(retries - 1)
        else:
            raise ValueError("Reached max number of retries")
    return page

if __name__ == '__main__':
    scrape_wiki()
