__authors__ = '1670849'
__group__ = 'TO_BE_FILLED'

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
        self.X = np.reshape(X, (np.prod(X.shape[:-1]), X.shape[-1]))

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
            self.centroids = np.random.rand(self.K, self.X.shape[1])

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
        sums = np.zeros((self.K, self.X.shape[-1]), dtype=np.float64)
        counts = np.zeros(self.K, dtype=np.long)

        for i in range(self.X.shape[0]):
            centroid = self.labels[i]
            sums[centroid] += self.X[i].astype(np.float64)
            counts[centroid] += 1

        counts = counts.astype(np.float64).repeat(self.X.shape[-1]).reshape(sums.shape)
        self.centroids = sums / counts

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
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
            if self.converges(): break

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass


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

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    P = X.shape[0]
    K = C.shape[0]

    dists = np.empty((P, K), dtype=np.float64)
    for i in range(P):
        for j in range(K):
            tmp = X[i].astype(np.float64) - C[j].astype(np.float64)
            dists[i][j] = np.sqrt(np.sum(tmp * tmp, dtype=np.float64), dtype=np.float64)

    return dists


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors)
