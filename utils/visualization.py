from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def tSNE(X, Y, name):
    X_embedded = TSNE(n_components=2).fit_transform(X)
    plt.figure()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y, s=1)
    plt.savefig("./Figures/ImageNet_dogs_100_NMI_0.275.png")
    plt.show()
    plt.close()