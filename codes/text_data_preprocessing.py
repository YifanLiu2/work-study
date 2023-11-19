import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


def plot_svd_performance(texts, n_components):
    """
    Plot the explained variance ratio vs. n_components.

    Parameters:
    texts (list of str): The list of texts to be processed.
    """
    vectorizor = TfidfVectorizer()
    tfidf_matrix = vectorizor.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(tfidf_matrix)
    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(svd.explained_variance_ratio_) + 1),
             np.cumsum(svd.explained_variance_ratio_), marker='o')
    plt.show()


def texts_vectorization(texts, n_components=100):
    """
    Preprocess the text data: Vectorize using TF-IDF and reduce dimensionality.

    Parameters:
    texts (list of str): The list of texts to be processed.
    n_components (int): The number of dimensions to reduce to using TruncatedSVD.

    Returns:
    array: The reduced dimensionality array of the TF-IDF vectors.
    """
    vectorizor = TfidfVectorizer()
    tfidf_matrix = vectorizor.fit_transform(texts)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_data = svd.fit_transform(tfidf_matrix)

    return reduced_data
