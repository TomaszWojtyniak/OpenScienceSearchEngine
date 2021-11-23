from __future__ import print_function

import pandas as pd
import tika
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram
from tika import parser
from summa import keywords


def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    words = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    for i in stems:
        if len(i) > 2:
            words.append(i)
    return words


def findKeywords(text):
    stemmer = SnowballStemmer("english")
    list_of_keywords = []
    words = []
    kWords = keywords.keywords(text).split("\n")
    for token in kWords:
        if re.search('[a-zA-Z]', token):
            list_of_keywords.append(token)
    stems = [stemmer.stem(t) for t in list_of_keywords]
    for i in stems:
        if len(i) > 2:
            words.append(i)
    print(words)
    return words


def main():
    tika.initVM()

    synopses = []

    publikacje = []
    for i in range(1, 2):  # 122
        if i <= 10:
            publikacje.append("architektura")
        if i > 10 and i <= 20:
            publikacje.append("ekonomia")
        if i > 20 and i <= 30:
            publikacje.append("matematyka")
        if i > 30 and i <= 40:
            publikacje.append("biologia")
        if i > 40 and i <= 60:
            publikacje.append("muzyka")
        if i > 60 and i <= 70:
            publikacje.append("architektura")
        if i > 70 and i <= 80:
            publikacje.append("ekonomia")
        if i > 80 and i <= 90:
            publikacje.append("biologia")
        if i > 90 and i <= 110:
            publikacje.append("chemia")
        if i > 110:
            publikacje.append("fizyka")

    for i in range(1, 7):  # 122
        item = parser.from_file('publikacje/' + str(i) + '.pdf')
        print(i)
        synopses.append(item['content'])

    tfidf_vectorizer = TfidfVectorizer(max_df=0.6, max_features=20000,
                                       min_df=0.4, stop_words='english',
                                       use_idf=True, tokenizer=findKeywords, ngram_range=(1, 8))
    tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

    print(type(tfidf_matrix))

    feature_names = tfidf_vectorizer.get_feature_names()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    df = pd.DataFrame(denselist, columns=feature_names)

    print(df)

    dist = 1 - cosine_similarity(tfidf_matrix)

    linkage_matrix = ward(dist)  # define the linkage_matrix using ward clustering pre-computed distances

    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=publikacje)

    plt.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout

    plt.show()
    plt.savefig('ward_clusters.png', dpi=200)  # save figure as ward_clusters


main()
