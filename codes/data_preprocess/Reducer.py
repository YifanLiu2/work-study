from sklearn.decomposition import TruncatedSVD

class Reducer:
    """
    Class for dimensionality reduction using Truncated SVD.
    """

    def __init__(self, n_components=100):
        """
        Initializes the reducer with a specified number of components.

        Parameters:
        n_components (int): The number of dimensions to reduce to.
        """
        self.reducer = TruncatedSVD(n_components=n_components, random_state=42)

    def reduce(self, feature_matrix):
        """
        Reduces the dimensionality of the given feature matrix.

        Parameters:
        feature_matrix: The feature matrix to be reduced.

        Returns:
        ndarray: The reduced dimensionality matrix.
        """
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
