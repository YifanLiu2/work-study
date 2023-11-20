from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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
        pass

class BoWVectorizer(BaseVectorizer):
    """
    Bag-of-Words vectorizer class.
    """

    def __init__(self, ngram_range=(1, 1)):
        """
        Initializes the BoW vectorizer with specified ngram range.

        Parameters:
        ngram_range (tuple): The ngram range for the vectorizer.
        """
        self.vectorizer = CountVectorizer(ngram_range=ngram_range)

    def vectorize(self, texts):
        """
        Vectorizes the given texts using Bag-of-Words model.

        Parameters:
        texts (list of str): The list of texts to be vectorized.

        Returns:
        Sparse matrix: The vectorized representation of texts.
        """
        return self.vectorizer.fit_transform(texts)

class TfidfVectorizer(BaseVectorizer):
    """
    TF-IDF vectorizer class.
    """

    def __init__(self, ngram_range=(1, 1)):
        """
        Initializes the TF-IDF vectorizer with specified ngram range.

        Parameters:
        ngram_range (tuple): The ngram range for the vectorizer.
        """
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)

    def vectorize(self, texts):
        """
        Vectorizes the given texts using TF-IDF model.

        Parameters:
        texts (list of str): The list of texts to be vectorized.

        Returns:
        Sparse matrix: The vectorized representation of texts.
        """
        return self.vectorizer.fit_transform(texts)
