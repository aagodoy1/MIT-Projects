"""Mixture model for collaborative filtering"""
from typing import NamedTuple, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc


class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component


def init(X: np.ndarray, K: int,
         seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """Initializes the mixture model with random points as initial
    means and uniform assingments

    Args:
        X: (n, d) array holding the data
        K: number of components
        seed: random seed

    Returns:
        mixture: the initialized gaussian mixture
        post: (n, K) array holding the soft counts
            for all components for all examples

    """
    np.random.seed(seed)
    n, _ = X.shape
    p = np.ones(K) / K

    # select K random points as initial means
    mu = X[np.random.choice(n, K, replace=False)]
    var = np.zeros(K)
    # Compute variance
    for j in range(K):
        var[j] = ((X - mu[j])**2).mean()

    mixture = GaussianMixture(mu, var, p)
    post = np.ones((n, K)) / K

    return mixture, post


def plot(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray,
         title: str):
    """Plots the mixture model for 2D data"""
    _, K = post.shape

    percent = post / post.sum(axis=1).reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    r = 0.25
    color = ["r", "b", "k", "y", "m", "c"]
    for i, point in enumerate(X):
        theta = 0
        for j in range(K):
            offset = percent[i, j] * 360
            # arc = Arc(point,
            #           r,
            #           r,
            #           0,
            #           theta,
            #           theta + offset,
            #           edgecolor=color[j])
            arc = Arc(point,
                        r,
                        r,
                        angle=0,
                        theta1=theta,
                        theta2=theta + offset,
                        edgecolor=color[j])
            ax.add_patch(arc)
            theta += offset
    for j in range(K):
        mu = mixture.mu[j]
        sigma = np.sqrt(mixture.var[j])
        circle = Circle(mu, sigma, color=color[j], fill=False)
        ax.add_patch(circle)
        legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(
            mu[0], mu[1], sigma)
        ax.text(mu[0], mu[1], legend)
    plt.axis('equal')
    plt.show()


def plot_multi(X, mixtures_dict, posts_dict, titles_dict):
    """Dibuja múltiples clusters en una figura con subplots sin tocar `plot()`."""
    K_vals = list(mixtures_dict.keys())
    num_plots = len(K_vals)

    n_rows = 2
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()

    color = ["r", "b", "k", "y", "m", "c"]
    r = 0.25

    for idx, K in enumerate(K_vals):
        ax = axes[idx]
        mixture = mixtures_dict[K]
        post = posts_dict[K]
        title = titles_dict[K]

        _, K_comp = post.shape
        percent = post / post.sum(axis=1).reshape(-1, 1)

        ax.set_title(title)
        # Calcular límites del gráfico automáticamente
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        y_min, y_max = X[:, 1].min(), X[:, 1].max()
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1

        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_aspect('equal')

        for i, point in enumerate(X):
            theta = 0
            for j in range(K_comp):
                offset = percent[i, j] * 360
                arc = Arc(point,
                          r,
                          r,
                          angle=0,
                          theta1=theta,
                          theta2=theta + offset,
                          edgecolor=color[j])
                ax.add_patch(arc)
                theta += offset

        for j in range(K_comp):
            mu = mixture.mu[j]
            sigma = np.sqrt(mixture.var[j])
            circle = Circle(mu, sigma, color=color[j], fill=False)
            ax.add_patch(circle)
            legend = "mu = ({:.2f}, {:.2f})\nstdv = {:.2f}".format(mu[0], mu[1], sigma)
            ax.text(mu[0], mu[1], legend)

    # Elimina ejes sobrantes si hay menos de 4
    for idx in range(num_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()
    
def rmse(X, Y):
    return np.sqrt(np.mean((X - Y)**2))

def bic(X: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians

    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data

    Returns:
        float: the BIC for this mixture
    """
    ## Ecuacion principal: 
    ## BCI = l - 1/2 * p * ln(n)
    ## Where 
    #   l: log verosimilitud del modelo
    #   p: N° parametros libres
    #   n: Cantidad de datos.

    # Part a) Calculate l
    n, d = X.shape
    K, _ = mixture.mu.shape
    
    #
    bic_value = log_likelihood - 0.5 * (K * d + 2 * K - 1) * np.log(n)

    return bic_value
    raise NotImplementedError
