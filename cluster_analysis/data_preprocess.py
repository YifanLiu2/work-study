from abc import ABC, abstractmethod
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse

class BaseVectorizer(ABC):
    """
    Abstract base class for text vectorization.
    """

    @abstractmethod
    def vectorize(self, texts):
        """
        Abstract method to vectorize given texts.

        Parameters:
        texts (list of str): The list of texts to be vectorized.

        Returns:
        The vectorized representation of texts.
        """
        raise NotImplementedError

class BoWVectorizer(BaseVectorizer):
    """
    Bag-of-Words vectorizer class.
    """

    def __init__(self, ngram_range=(1, 1)):
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)

    def vectorize(self, texts):
        return self.vectorizer.fit_transform(texts)

class TfidfVectorizer(BaseVectorizer):
    """
    TF-IDF vectorizer class.
    """

    def __init__(self, ngram_range=(1, 1)):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)

    def vectorize(self, texts):
        return self.vectorizer.fit_transform(texts)

class CharacterLevelVectorizer(BaseVectorizer):
    """
    Character-level vectorizer class.
    """

    def __init__(self):
        self.char_to_index = None

    def vectorize(self, texts):
        if not self.char_to_index:
            self.set_char_to_index
        
        encoded_sequences = []
        for text in texts:
            encoded_text = [self.char_to_index.get(char, 0) for char in text]
            encoded_sequences.append(encoded_text)
            self.max_length = max(self.max_length, len(encoded_text))
        padded_sequences = [self._pad_sequence(seq) for seq in encoded_sequences]
        matrix = sparse.csr_matrix(padded_sequences)
        return matrix
    
    def set_char_to_index(self, texts):
        """
        Create a mapping from characters to indices based on the provided texts.
        
        Parameters:
        texts (list of str): The list of texts to create the mapping from.
        """
        all_text = ''.join(texts)
        unique_chars = sorted(set(all_text))
        self.char_to_index = {char: idx + 1 for idx, char in enumerate(unique_chars)}

    def _pad_sequence(self, sequence):
        """
        Pad the sequence to ensure uniform length.

        Parameters:
        sequence (list of int): The character-encoded sequence to pad.

        Returns:
        list of int: The padded sequence.
        """
        padding = [0] * (self.max_length - len(sequence))
        return sequence + padding


class Reducer:
    """
    Class for dimensionality reduction using Truncated SVD.
    """

    def __init__(self, n_components=100):
        self.reducer = TruncatedSVD(n_components=n_components, random_state=42)

    def reduce(self, feature_matrix):
        return self.reducer.fit_transform(feature_matrix)

    def plot_variance_explained(self):
        """
        Plots the cumulative explained variance ratio vs. number of components.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.reducer.explained_variance_ratio_), marker='o')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.show()


class TextProcessor:
    """
    Class for processing text data using vectorization and optional dimensionality reduction.
    """

    def __init__(self, vectorizer, reducer=None):
        self.vectorizer = vectorizer
        self.reducer = reducer

    def process(self, texts):
        """
        Processes the given texts through vectorization and optional reduction.

        Parameters:
        texts (list of str): The list of texts to be processed.

        Returns:
        The processed feature matrix.
        """
        features = self.vectorizer.vectorize(texts)
        if self.reducer:
            features = self.reducer.reduce(features)
        return features

    def plot_variance_explained(self):
        """
        Plots the cumulative explained variance ratio for dimensionality reduction.
        Raises an error if reducer is not defined.
        """
        if not self.reducer:
            raise ValueError("Reducer not defined.")
        self.reducer.plot_variance_explained()
