"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    post_cust_cluster = np.zeros((n, K))
    epsilon = 10**-16
    log_likelihood = 0

    #0) Calcular las medias

    # mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    # var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    # p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component

    # 1) calcular log-posteriori (primero necesitamos p de cluster dado user)
    for customer in range(n):
        func_cust_cluster = np.zeros(K)

        #X_nan = np.where(X[customer] == 0, np.nan, X[customer])
        indexes = np.where(X[customer] != 0)[0]
        for cluster in range(K):
            x_cust_pel = X[customer, indexes]
            mean_clust_pel = mixture.mu[cluster, indexes]
            sigma2_cluster = mixture.var[cluster]
            p_cust_cluster = mixture.p[cluster]

    # 1.2) calcular log-posteriori (# Log de la densidad gaussiana multivariada isotrópica:)
            log_N = -len(indexes) * 0.5 * np.log(2*np.pi*sigma2_cluster) - (1/(2*sigma2_cluster)) * np.sum((x_cust_pel-mean_clust_pel)**2)
            func_cust_cluster[cluster] = np.log(p_cust_cluster + epsilon) + log_N
        
        logsum = logsumexp(func_cust_cluster)

        for cluster in range(K):
            log_post_cust_cluster = func_cust_cluster[cluster] - logsum
            post_cust_cluster[customer, cluster] = np.exp(log_post_cust_cluster)
        log_likelihood += logsum

    return post_cust_cluster, log_likelihood
    raise NotImplementedError



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    n, d = X.shape
    K, _ = mixture.mu.shape

    mu = np.zeros((K, d))
    var = np.zeros(K)
    p = np.zeros(K)

    N_cluster = 0 
    # mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    # var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    # p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component

    # for cluster in range(K):
    #     N_cluster = 0
    #     N_cluster += np.sum(post, axis = 0)

    N_K = np.sum(post, axis = 0) # Peso total del cluster
    p = N_K/n # Peso del cluster asociado a cada n

    for cluster in range(K):

        for pelicula in range(d):

            numerador = 0
            denominador = 0

            for customer in range(n):

                if X[customer, pelicula] != 0:
                #if full_points[customer] and X[customer, pelicula] != 0:
                    numerador += post[customer, cluster] * X[customer, pelicula]
                    denominador += post[customer, cluster]
            
            #if denominador > 0:
            if denominador >= 1-1e-8:
                mu[cluster, pelicula] = numerador / denominador
            else:
                #mu[cluster, pelicula] = 0
                mu[cluster, pelicula] = mixture.mu[cluster, pelicula]

    for cluster in range(K):
        numerador = 0
        denominador = 0
        for customer in range(n):
            indexes = np.where(X[customer] != 0)[0] #
            if len(indexes) == 0:
                continue  # skip users with no data
            diff = (X[customer, indexes]-mu[cluster, indexes])
            error = np.sum(diff**2)

            numerador += post[customer, cluster] * error
            denominador += post[customer, cluster] * len(indexes)

        #var[cluster] = numerador / denominador

        #if denominador >= 1:
        if denominador > 1e-8:
            var[cluster] = numerador / denominador
        else:
            var[cluster] = min_variance

        if var[cluster] < min_variance:
            var[cluster] = min_variance
    
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
