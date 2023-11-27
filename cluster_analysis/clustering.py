from abc import ABC, abstractmethod
import numpy as np
from sklearn.model_selection import ParameterSampler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


class BaseClusterer(ABC):
    """
    Abstract class for clustering
    """
    def __init__(self, k):
        """
        """
        self.k = k
    
    @abstractmethod
    def clustering(self, data, tune=False, n_iter=10):
        """
        Abstract method for clustering.

        Parameters:
        data (array-like): Data points for clustering, where each data point is a feature vector.

        Returns:
        array: An array of cluster labels indicating the cluster each data point belongs to.
        """
        raise NotImplementedError
    
    def tune_parameters(self, data, n_iter, param_distributions):
        """
        Tune parameters of the clustering model.

        Parameters:
        data (array-like): Data points for clustering.
        n_iter (int): Number of iterations for parameter tuning.
        param_distributions (dict): Parameter distribution for sampling.
        """
        best_score = -np.inf
        best_params = None

        for params in ParameterSampler(param_distributions, n_iter, random_state=42):
            try:
                self.clusterer.set_params(**params)
                self.clusterer.fit(data)
                score = silhouette_scorer(self.clusterer, data)

                if score > best_score:
                    best_score = score
                    best_params = params
            except ValueError:
                continue

        print(f"Best Params: {best_params}, Best Score: {best_score}")
        self.clusterer.set_params(**best_params)

class KMeansClusterer(BaseClusterer):
    """
    A clustering algorithm based on the K-Means algorithm.
    """
    def __init__(self, k):
        super().__init__(k)
        self.clusterer = KMeans(n_clusters=self.k, n_init=10, random_state=42)
    
    def clustering(self, data, tune=False, n_iter=10):
        if tune:
            param_distributions = {
                'n_clusters': [self.k],
                'init': ['k-means++', 'random'],
                'n_init': [10, 15, 20, 25, 30],
                'max_iter': [100, 200, 300, 400, 500],
                'random_state': [42],
            }
            self.tune_parameters(data, n_iter, param_distributions)

        self.clusterer.fit(data)
        return self.clusterer.labels_

class HierarchicalClusterer(BaseClusterer):
    """
    A clustering algorithm based on hierarchical/agglomerative clustering.
    """
    def __init__(self, k):
        super().__init__(k)
        self.clusterer = AgglomerativeClustering(n_clusters=self.k)
    
    def clustering(self, data, tune=False, n_iter=10):
        if tune:
            param_distributions = {
                'n_clusters': [self.k],
                'metric': ['euclidean', 'manhattan', 'cosine'],
                'linkage': ['ward', 'complete', 'average', 'single'],
            }
            self.tune_parameters(data, n_iter, param_distributions)

        self.clusterer.fit(data)
        return self.clusterer.labels_
    
    def get_ordering(self):
        """
        Retrieves the order of merges in the hierarchical clustering process.

        Returns:
        numpy.ndarray: An array where each row represents a merge operation. Each row
                       contains two indices, indicating the clusters/points that were merged.
        """
        if not hasattr(self.clusterer, 'children_'):
            raise ValueError("The clustering model must be fit before getting the merge order.")

        # The order of merges is directly given by 'children_'
        merge_order = self.clusterer.children_
        return merge_order


class GMMClusterer(BaseClusterer):
    """
    A clustering algorithm based on the Gaussian Mixture Model.
    """
    def __init__(self, k):
        super().__init__(k)
        self.clusterer = GaussianMixture(n_components=k, random_state=42)
    
    def clustering(self, data, tune=False, n_iter=10):
        if tune: 
            param_distributions = {
                'n_components': [self.k],
                'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                'tol': [1e-3, 1e-4, 1e-5],
                'max_iter': [100, 200, 300, 400, 500],
                'init_params': ['kmeans', 'random'],
                'random_state': [42]
            }
            self.tune_parameters(data, n_iter, param_distributions)

        self.clusterer.fit(data)
        return self.clusterer.predict(data)
    

def silhouette_scorer(estimator, X):
    """
    Custom scorer function that uses the silhouette score.
    This function is meant to be used as a scorer compatible with scikit-learn's
    model selection and evaluation routines.

    Parameters:
        estimator: The clustering model to evaluate.
        X (array-like): Data points used for clustering.

    Returns:
        float: The silhouette score.
    """
    if hasattr(estimator, 'labels_'):
        clusters = estimator.labels_
    elif hasattr(estimator, 'predict'):
        estimator.fit(X)
        clusters = estimator.predict(X)
    else:
        raise ValueError("The estimator must have a 'labels_' attribute or a 'predict' method")
    return silhouette_score(X, clusters)

    
