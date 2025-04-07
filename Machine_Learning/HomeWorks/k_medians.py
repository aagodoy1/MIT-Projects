import numpy as np
from sklearn.metrics import pairwise_distances

def kmedians(X, k, max_iter=100, init=None):
    """
    K-medians clustering (basado en distancia L1).
    
    Parámetros:
    - X: np.array, datos
    - k: int, número de clusters
    - max_iter: int, número máximo de iteraciones
    - init: None o lista/array de índices o valores para los centros iniciales
    """
    X = np.asarray(X)

    # Inicializar medians
    if init is not None:
        if isinstance(init[0], int):  # si son índices
            medians = X[init]
        else:  # si son vectores
            medians = np.array(init)
    else:
        rng = np.random.default_rng()
        medians = X[rng.choice(len(X), k, replace=False)]

    for _ in range(max_iter):
        distances = pairwise_distances(X, medians, metric='manhattan')
        labels = np.argmin(distances, axis=1)

        new_medians = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                new_medians.append(medians[i])
            else:
                new_medians.append(np.median(cluster_points, axis=0))
        new_medians = np.array(new_medians)

        if np.allclose(medians, new_medians):
            break
        medians = new_medians

    return labels, medians
