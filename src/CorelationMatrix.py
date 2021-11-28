from scipy.cluster.hierarchy import ward
from sklearn.metrics.pairwise import cosine_similarity
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer

def getSynopses(pdfs):
    synopses = []
    for i in range(1, pdfs):
        save_path = '/src/publikacjeRaw'
        name_of_file = str(i)
        completeName = os.path.join(save_path, name_of_file + ".txt")
        with open(completeName, 'r') as f:
            synopses.append(f.read())
        f.close()
    return synopses

def getUniqueWords():
    save_path = '/src/uniqueWords'
    name_of_file = "../files/uniqueWords"
    completeName = os.path.join(save_path, name_of_file + ".txt")
    with open(completeName, 'r') as f:
        uniqueWords = f.read().split()
    f.close()
    return uniqueWords


def get_similarity_matrix(synopses):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.6, max_features=200000, min_df=0.3,
                                       stop_words='english',use_idf=True, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return ward(similarity_matrix)