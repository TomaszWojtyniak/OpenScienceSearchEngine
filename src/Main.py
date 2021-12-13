from CorelationMatrix import *
from DocumentsCleaning import *
from TextProcessing import *
from Plot import *
import nltk

PDFs = 120 #122

if __name__ == "__main__":
    stopwords = nltk.corpus.stopwords.words('english')
    publikacje = getCategoriesNames(PDFs)

    #text = getText(PDFs)
    #text = getTextFromTxt(PDFs)
    #text = preprocess(text)
    #saveProcessedText(text)
    text = getProcessedText(PDFs)
    linkage_matrix = get_similarity_matrix(text)
    getSimilarityRanking(linkage_matrix, publikacje)
    #doDendogram(linkage_matrix, publikacje)