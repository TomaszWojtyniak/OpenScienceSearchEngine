import nltk
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string


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
    #lemm_text = " ".join(lemm_text)
    return " ".join(lemm_text)

def removeNotWantedWords(text):
    text = [i for i in text if i.isalpha()]
    text = [i for i in text if len(i) > 3 and len(i) < 12]
    return text

def preprocess(pdfs):
    list = []
    for data in pdfs:
        data = removePunctuation(data)
        data = lowerText(data)
        data = tokenization(data)
        data = remove_stopwords(data)
        data = removeNotWantedWords(data)
        data = stemming(data)
        data = lemmatizer(data)
        list.append(data)
    return list