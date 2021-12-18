import numpy
from scipy.cluster.hierarchy import ward, fcluster
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

def get_similarity_matrix(synopses):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.6, max_features=200000, min_df=0.3,
                                       stop_words='english',use_idf=True, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    w = ward(similarity_matrix)
    return ward(similarity_matrix)

def getSimilarityRanking(matrix, publikacje):
    threshold = 0.01
    ranking = fcluster(matrix, threshold, criterion='distance')
    rankingDictionary = {i: [] for i in range(0, len(ranking))}
    while max(ranking) != 1:
        for index, value in enumerate(ranking):
            similarities = numpy.where(ranking == value)[0]
            for element in similarities:
                if element not in rankingDictionary[index] and element != index:
                    rankingDictionary[index].append(element)
        threshold = threshold + 0.01
        ranking = fcluster(matrix, threshold, criterion='distance')


    for article in range(0,len(matrix)):
        save_path = '../files/ranking'
        name_of_file = str(article)
        completeName = os.path.join(save_path, name_of_file + ".txt")

        ranking = rankingDictionary[article]
        similarityResult = []
        for number in ranking:
            similarityResult.append(publikacje[number])
        with open(completeName, 'w') as f:
            f.write(str(similarityResult))
        f.close()


