from abc import ABC, abstractmethod
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV


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
    
    def tune_parameters(self, data, n_iter, params):
        """
        Tune parameters of the clustering model using RandomizedSearchCV.

        Parameters:
        data (array-like): Data points for clustering.
        n_iter (int): Number of iterations for parameter tuning.
        param_distributions (dict): Parameter distribution for RandomizedSearchCV.
        """
        random_search = RandomizedSearchCV(
            estimator=self.clusterer,
            param_distributions=params,
            n_iter=n_iter,
            scoring=make_scorer(silhouette_scorer, greater_is_better=True),
            random_state=42
        )
        search_result = random_search.fit(data)
        print(f"Best Params: {search_result.best_params_}, Best Score: {search_result.best_score_}")
        self.clusterer = search_result.best_estimator_

class KMeansClusterer(BaseClusterer):
    """
    A clustering algorithm based on the K-Means algorithm.
    """
    def __init__(self, k):
        super().__init__(k)
        self.clusterer = KMeans(n_clusters=self.k, random_state=42)
    
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
                'affinity': ['euclidean', 'manhattan', 'cosine'],
                'linkage': ['ward', 'complete', 'average', 'single'],
            }
            self.tune_parameters(data, n_iter, param_distributions)

        self.clusterer.fit(data)
        return self.clusterer.labels_
    
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

    Parameters:
        estimator: The clustering model to evaluate.
        X (array-like): Data points used for clustering.

    Returns:
        float: The silhouette score.
    """
    if hasattr(estimator, 'labels_'):
        estimator.fit(X)
        clusters = estimator.labels_
    elif hasattr(estimator, 'predict'):
        clusters = estimator.fit_predict(X)
    else:
        raise ValueError("The estimator must have a 'labels_' attribute or a 'predict' method")
    score = silhouette_score(X, clusters)
    return score
    
