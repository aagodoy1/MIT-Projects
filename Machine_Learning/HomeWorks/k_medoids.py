import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils.validation import check_array

class KMedoids(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, max_iter=300, random_state=None,
                 metric='euclidean', init='random'):
        """
        init: 'random' o lista de índices para inicializar los medoids
        metric: 'euclidean' (L2) o 'manhattan' (L1)
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.metric = metric
        self.init = init

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples = X.shape[0]

        # Inicialización de los medoids
        if isinstance(self.init, list) or isinstance(self.init, np.ndarray):
            medoid_indices = np.array(self.init)
        elif self.init == 'random':
            rng = np.random.RandomState(self.random_state)
            medoid_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        else:
            raise ValueError("init debe ser 'random' o una lista de índices.")

        medoids = X[medoid_indices]

        for _ in range(self.max_iter):
            # Asignar puntos al medoid más cercano
            distances = self._pairwise_distances(X, medoids)
            labels = np.argmin(distances, axis=1)

            new_medoids = np.copy(medoids)
            for k in range(self.n_clusters):
                cluster_k = X[labels == k]
                if len(cluster_k) == 0:
                    continue
                distance_matrix = self._pairwise_distances(cluster_k, cluster_k)
                medoid_idx = np.argmin(np.sum(distance_matrix, axis=1))
                new_medoids[k] = cluster_k[medoid_idx]

            if np.allclose(medoids, new_medoids):
                break
            medoids = new_medoids

        self.cluster_centers_ = medoids
        self.labels_ = np.argmin(self._pairwise_distances(X, medoids), axis=1)
        return self

    def _pairwise_distances(self, X1, X2):
        if self.metric == 'euclidean':
            return np.linalg.norm(X1[:, np.newaxis] - X2, axis=2)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X1[:, np.newaxis] - X2), axis=2)
        else:
            raise ValueError("Solo se permiten métricas 'euclidean' o 'manhattan'.")
