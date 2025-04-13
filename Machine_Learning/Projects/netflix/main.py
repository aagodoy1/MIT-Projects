import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
'''
# PREGUNTA 1)
Ks = [1,2,3,4]
seeds = [0,1,2,3,4]


# Diccionario para guardar el mejor costo y el mejor mezcla por cada K
best_costs = {}
best_mixtures = {}

for K in Ks:
    print(f'Starting K = {K}')
    lowest_cost = float('inf')
    best_mix = None

    for seed in seeds:
        print(f'Staring seed {seed}')
        # Inicializar mezcla para K-means
        mixture, post = common.init(X, K, seed)
        #print(f'mixture = {mixture} and post = {post.ndim}')

        # Ejecutar K-means
        mixture, post, cost = kmeans.run(X, mixture, post)
        #print(f'for kmeans, mixture = {mixture}, post = {post.ndim}, cost = {cost}')

        # Verificar si es el mejor costo encontrado hasta ahora para este K
        if cost < lowest_cost:
            lowest_cost = cost
            best_mix = mixture

    # Guardar el mejor costo y mezcla
    best_costs[K] = lowest_cost
    best_mixtures[K] = best_mix

    # Graficar la mejor solución para este K
    common.plot(X, best_mix, post, f"K={K}")

# Imprimir resultados
for K in Ks:
    print(f"Cost|K={K} = {best_costs[K]}")
'''
# PREGUNTA 2

k = 3
seed = 0
mixture, post = common.init(X, k, seed)

mixture, post, cost = naive_em.run(X, mixture, post)
