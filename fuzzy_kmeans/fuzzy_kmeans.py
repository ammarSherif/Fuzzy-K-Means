"""Fuzzy K-means clustering"""

# ==============================================================================
# Author: Ammar Sherif <ammarsherif90 [at] gmail [dot] com >
# ==============================================================================

# ==============================================================================
# The file includes an implementation of a fuzzy version of kmeans with sklearn-
# like interface.
# ==============================================================================

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state

class FuzzyKMeans(KMeans):
    """The class implements the fuzzy version of kmeans
    ----------------------------------------------------------------------------
    Args: same arguments as in SKlearn in addition to
        - m: the fuzziness index to determine how fuzzy our boundary is
        - eps: the tolerance value for convergence
    """
    def __init__(self, m, eps= 0.001,*args, **kwargs):
        self.__m = m
        self.__eps = eps
        super(FuzzyKMeans, self).__init__(*args, **kwargs)

    # --------------------------------------------------------------------------

    def _check_params(self, X):
        if (self.__m <= 1):
            raise ValueError(
                "the fuzziness index m should be more than 1"
                f", got '{self.__m}' instead."
            )
        super(FuzzyKMeans, self)._check_params(X)

    # --------------------------------------------------------------------------

    def __compute_dist(self, data, centroids):
        """The method computes the distance matrix for each data point with res-
        pect to each cluster centroid.
        ------------------------------------------------------------------------

        Inputs:
            - data: the input data points
            - centroids: the clusters' centroids

        Output:
            - distance_m: the distace matrix
        """
        n_points = data.shape[0]
        n_clusters = centroids.shape[0]

        distance_m = np.zeros((n_points, n_clusters))

        for i in range(n_clusters):
            diff = X-centroids[i,:]
            distance_m[:,i] = np.sqrt((diff * diff).sum(axis=1))

        return distance_m

    # --------------------------------------------------------------------------

    def compute_membership(self, data, centroids):
        """The method computes the  membership matrix  of the data  according to
        the clusters specified by the given centroids

        Inputs:
            - data: the input data points being clustered
            - centroids: numpy array including the cluster centroids;
                its shape is (n_clusters, n_features)

        Outputs:
            - fmm: fuzzy membership matrix"""
        # ----------------------------------------------------------------------
        # First, compute  the distance between the point and the other centroids
        # ======================================================================
        # Note we also add alpha,  1e-10 very little value, as  we are computing
        # 1 over the distances, and there might be 0 distance
        # ----------------------------------------------------------------------

        dist = self.__compute_dist(data, centroids) + 1e-10
        # ----------------------------------------------------------------------
        # We are computing the below value once  because we need it  in both the
        # numerator and the denominator of the value to be computed
        # ----------------------------------------------------------------------
        sqr_dist = dist**(-2/(self.__m-1))

        # ----------------------------------------------------------------------
        # We compute the normalizing term (denominator)
        # ----------------------------------------------------------------------
        norm_dist = np.expand_dims(np.sum(sqr_dist,axis=1),axis=1)

        fmm = sqr_dist / norm_dist
        return fmm


    # --------------------------------------------------------------------------

    def fit(self, X, y=None, sample_weight=None):
        """The method computes the fuzzy  k-means clustering algorithm

        Inputs:
            - X: training data
            - y: ignored
            - sample_weight: weights of each data point
        """
        # ----------------------------------------------------------------------
        # Number of iterations
        # ----------------------------------------------------------------------
        i = 1
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        # ----------------------------------------------------------------------
        # Initialize the centroids
        # ----------------------------------------------------------------------

        centroids = self._init_centroids(X,x_squared_norms=None,init= self.init,
                                         random_state= random_state)
        print(centroids.shape)
        print(type(centroids))
        # ======================================================================
        # Do the first iteration
        # ======================================================================
        fmm = self.compute_membership(X, centroids)
