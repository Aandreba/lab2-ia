__authors__ = ['1670849']
__group__ = '83'

import numpy as np
import utils

class KMeans:
    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        self.X = np.reshape(X, (np.prod(X.shape[:-1]), X.shape[-1])).astype(np.float64)

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        self.old_centroids = np.zeros((self.K, self.X.shape[1]))
        if self.options['km_init'].lower() == 'first':
            self.centroids = np.array([self.X[0]])
            for i in range(1, self.X.shape[0]):
                if not (self.centroids == self.X[i]).all(1).any():
                    self.centroids = np.append(self.centroids, [self.X[i]], axis=0)
                    if self.centroids.shape[0] == self.K:
                        break
        else:
            min = np.min(self.X)
            max = np.max(self.X)
            self.centroids = (max - min) * np.random.rand(self.K, self.X.shape[1]) + min

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        dists = distance(self.X, self.centroids)
        self.labels = np.argmin(dists, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        self.old_centroids = self.centroids
        self.centroids = np.empty((self.K, self.X.shape[-1]), dtype=np.float64)
        for i in range(self.K):
            self.centroids[i] = np.mean(self.X[self.labels == i], axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        return np.allclose(self.centroids, self.old_centroids, equal_nan=True)
     
    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        self._init_centroids()
        for i in range(100 if self.options["max_iter"] else self.options["max_iter"]):
            self.get_labels()
            self.get_centroids()
            if self.converges(): return

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        self.fit()
        return np.mean((self.X - self.centroids[self.labels])**2)

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        self.K = 1
        prev_wcd = self.withinClassDistance()        
        for i in range(2, max_K+1):
            self.K = i
            wcd = self.withinClassDistance()
            dec_pct = wcd / prev_wcd
            if (1.0 - dec_pct) <= 0.2:
                self.K = i - 1
                return
            prev_wcd = wcd

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    deltas = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    max_probs = np.argmax(utils.get_color_prob(centroids), axis=1)
    return list(utils.colors[max_probs])
