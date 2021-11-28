from CorelationMatrix import *
from DocumentsCleaning import *
from TextProcessing import *
from Plot import *
import nltk

PDFs = 122 #122

if __name__ == "__main__":
    stopwords = nltk.corpus.stopwords.words('english')
    publikacje = getCategoriesNames(PDFs)

    #text = getText(PDFs)
    text = getTextFromTxt(PDFs)
    print(text)
    text = preprocess(text)
    print(text)

    #synopses = getSynopses(pdfs)
    #uniqueWords = getUniqueWords()
    linkage_matrix = get_similarity_matrix(text)
    doDendogram(linkage_matrix, publikacje)