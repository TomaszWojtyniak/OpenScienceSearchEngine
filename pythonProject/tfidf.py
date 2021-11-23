import nltk
import numpy as np
from matplotlib import pyplot as plt
from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
from summa import keywords
from tika import parser
from num2words import num2words
import os.path
from nltk.stem import WordNetLemmatizer
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from scipy.cluster import  hierarchy
import string
import re

def removePunctuation(text):
    punctuationfree = "".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def lowerText(text):
    return text.lower()  

def tokenization(text):
    tokens = text.split(" ")
    return tokens

def remove_stopwords(text):
    stopwords = nltk.corpus.stopwords.words('english')
    output= [i for i in text if i not in stopwords]
    return output

def stemming(text):
    porter_stemmer = PorterStemmer()
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

def lemmatizer(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    lemm_text = " ".join(lemm_text)
    return lemm_text

def removeNotWantedWords(text):
    text = [i for i in text if i.isalpha()]
    text = [i for i in text if len(i) > 3 and len(i) < 12]
    return text

def preprocess(data):
    data = removePunctuation(data)
    data = lowerText(data)
    data = tokenization(data)
    data = remove_stopwords(data)
    data = removeNotWantedWords(data)
    data = stemming(data)
    data = lemmatizer(data)
    return data

def findKeywords(text):
    bagOfWords = []
    words = preprocess(text)
    words = keywords.keywords(words).split("\n")
    for i in words:
        if len(i) > 2 and len(i) < 20:
            bagOfWords.append(i)
    return bagOfWords

def getCategoriesNames():
    publikacje = []
    for i in range(1, 122):  # 122
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
    return publikacje


def getText():
    texts = []
    for i in range(1, 122):  # 122
        item = parser.from_file('publikacje/' + str(i) + '.pdf')
        texts.append(item['content'])

        save_path = '/Users/tomaszwojtyniakmoodupteam/PycharmProjects/Inzynierka/pythonProject/publikacjeRaw'
        name_of_file = str(i)
        completeName = os.path.join(save_path, name_of_file + ".txt")
        with open(completeName, 'w') as f:
            f.write(' '.join(item['content'].split()))
        f.close()
    return texts

def getTextFromTxt():
    texts = []
    for i in range(1, 122):
        save_path = '/Users/tomaszwojtyniakmoodupteam/PycharmProjects/Inzynierka/pythonProject/publikacjeRaw'
        name_of_file = str(i)
        completeName = os.path.join(save_path, name_of_file + ".txt")
        with open(completeName, 'r') as f:
            texts.append(f.read())
        f.close()
    return texts

def getKeywords(text):
    synopses = []
    uniqueWords = []
    nr = 1
    for i in text:
        print(nr)
        bag = findKeywords(i)
        synopses.append(" ".join(bag))
        uniqueWords = set(uniqueWords).union(set(bag))

        save_path = '/Users/tomaszwojtyniakmoodupteam/PycharmProjects/Inzynierka/pythonProject/keywords'
        name_of_file = str(nr)
        completeName = os.path.join(save_path, name_of_file + ".txt")
        with open(completeName, 'w') as f:
            f.write(' '.join(bag))
        f.close()
        nr = nr + 1
    save_path = '/Users/tomaszwojtyniakmoodupteam/PycharmProjects/Inzynierka/pythonProject/uniqueWords'
    name_of_file = "uniqueWords"
    completeName = os.path.join(save_path, name_of_file + ".txt")
    with open(completeName, 'w') as f:
        f.write(' '.join(uniqueWords))
    f.close()
    return synopses, uniqueWords

def getSynopses():
    synopses = []
    for i in range(1, 122):
        save_path = '/Users/tomaszwojtyniakmoodupteam/PycharmProjects/Inzynierka/pythonProject/publikacjeRaw'
        name_of_file = str(i)
        completeName = os.path.join(save_path, name_of_file + ".txt")
        with open(completeName, 'r') as f:
            synopses.append(f.read())
        f.close()
    return synopses

def getUniqueWords():
    save_path = '/Users/tomaszwojtyniakmoodupteam/PycharmProjects/Inzynierka/pythonProject/uniqueWords'
    name_of_file = "uniqueWords"
    completeName = os.path.join(save_path, name_of_file + ".txt")
    with open(completeName, 'r') as f:
        uniqueWords = f.read().split()
    f.close()
    return uniqueWords


def get_similarity_matrix(synopses):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.6, max_features=200000, min_df=0.3,
                                       stop_words='english',use_idf=True, ngram_range=(1,3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return ward(similarity_matrix)


def doDendogram(matrix, publikacje):
    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    ax = dendrogram(matrix, orientation="right", labels=publikacje)

    plt.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout

    plt.show()
    plt.savefig('ward_clusters.png', dpi=200)


def main():
    stopwords = nltk.corpus.stopwords.words('english')
    publikacje = getCategoriesNames()

    #text = getText()
    text = getTextFromTxt()
    #synopses, uniqueWords = getKeywords(text)

    #synopses = getSynopses()
    #uniqueWords = getUniqueWords()
    linkage_matrix = get_similarity_matrix(text)
    doDendogram(linkage_matrix, publikacje)
main()