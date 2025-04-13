import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# TODO: Your code here

# Inicialización
mixture, post = common.init(X, K, seed)

# Ejecutar EM
mixture, post, cost = em.run(X, mixture, post)

# Imputar datos faltantes con la mezcla aprendida
X_pred = common.fill_matrix(X, mixture)

# Evaluar el error entre la matriz completa predicha y la verdadera
mse = ((X_pred - X_gold) ** 2).sum() / np.count_nonzero(~np.isnan(X))

print(f"MSE entre X_gold y X_pred: {mse:.4f}")
print(f"Costo final del EM: {cost:.4f}")

