"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    #mu: np.ndarray  # (K, d)
    #var: np.ndarray  # (K, ) 
    #p: np.ndarray  # (K, ) 

    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    # 1) Calcular las densidades

    for k in range(K):
        diff = X-mixture.mu[k]
        sq_dist = np.sum(diff**2, axis = 1)
        expo = np.exp(-sq_dist/(2*mixture.var[k]))
        coef =  1 / ((2*np.pi*mixture.var[k])**(d/2))
        densities = coef*expo
        # 1.2) Unir la probabilidad priori con posteriori
        post[:,k] = mixture.p[k]*densities

    # 2) Calcular la probabilidad de la densidad
    suma_de_densidades = np.sum(post, axis = 1, keepdims = True)

    # 3) log_verosimilitud total
    log_L = np.sum(np.log(suma_de_densidades))

    return post/suma_de_densidades, log_L

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    mu = np.zeros((K, d))
    var = np.zeros(K)

    #for k in range(K):
    #    N_k = np.sum(post[k])
    #    mu_k = np.sum(post[k]*X, axis = 1)/N_k
    
    N_k = np.sum(post, axis = 0) #colapsa las filas en un array de k columnas (k,)
    pi_k = N_k / n
    N_k = N_k.reshape(K,1) #Se le da forma (k,1)
    N_k = np.tile(N_k, (1,d)) # le da dimensiones K,d repitiendo d veces las k columnas

    # Se hace producto punto de post.T = (K,n) y X que es (n,d) y el resultado será
    # de dimensiones K,d
    mu_k = (post.T @ X)/N_k  # Dimensiones (k,d)

    #X.shape == (n, d)
    #mu.shape == (K, d)
    #post.shape == (n, K)

    for k in range(K):
        diff = X-mu_k[k]
        suma = np.sum(diff**2, axis = 1)
        var_K = (post[:, k] @ suma)/(N_k[k,0]*d)
        var[k] = var_K

    return GaussianMixture(mu_k, var, pi_k)

    raise NotImplementedError


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
    prev_log_likehood = None
    log_likehood = None
    while (prev_log_likehood is None or  log_likehood - prev_log_likehood > 10**(-6) * abs(log_likehood)):
        prev_log_likehood = log_likehood
        post, log_likehood = estep(X, mixture)
        mixture = mstep(X, post)
    return mixture, post, log_likehood
    raise NotImplementedError
