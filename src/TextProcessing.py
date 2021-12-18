from tika import parser
import os.path

def getText(pdfs):
    texts = []
    for i in range(1, pdfs):  # 122
        item = parser.from_file('../files/publikacje/' + str(i) + '.pdf')
        texts.append(item['content'])

        save_path = '../files/publikacjeRaw'
        name_of_file = str(i)
        completeName = os.path.join(save_path, name_of_file + ".txt")
        with open(completeName, 'w') as f:
            f.write(' '.join(item['content'].split()))
        f.close()
    return texts

def getTextFromTxt(pdfs):
    texts = []
    for i in range(1, pdfs):
        save_path = '../files/publikacjeRaw'
        name_of_file = str(i)
        completeName = os.path.join(save_path, name_of_file + ".txt")
        with open(completeName, 'r') as f:
            texts.append(f.read())
        f.close()
    return texts

def saveProcessedText(pdfs):
    for i in range(1,len(pdfs)):
        save_path = '../files/processedText'
        name_of_file = str(i)
        completeName = os.path.join(save_path, name_of_file + ".txt")
        with open(completeName, 'w') as f:
            f.write(pdfs[i])
        f.close()

def getProcessedText(pdfs):
    texts = []
    for i in range(1, pdfs):
        save_path = '../files/processedText'
        name_of_file = str(i)
        completeName = os.path.join(save_path, name_of_file + ".txt")
        with open(completeName, 'r') as f:
            texts.append(f.read())
        f.close()
    return texts