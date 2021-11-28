from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def getCategoriesNames(pdfs):
    publikacje = []
    for i in range(1, pdfs):  # 122
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
    plt.savefig('../plots/ward_clusters.png', dpi=200)