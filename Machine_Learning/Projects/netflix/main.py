import numpy as np
import kmeans
import common
import naive_em
import em

import matplotlib.pyplot as plt

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
    #common.plot(X, best_mix, post, f"K={K}")

# Imprimir resultados
for K in Ks:
    print(f"Cost|K={K} = {best_costs[K]}")

# PREGUNTA 2


Ks = [1,2,3,4]
seeds = [0,1,2,3,4]


# Diccionario para guardar el mejor costo y el mejor mezcla por cada K
best_log_likehood = {}
best_mixtures = {}

for K in Ks:
    print(f'Starting K = {K}')
    highest_likehood = float('-inf')
    best_mix = None

    for seed in seeds:
        print(f'Staring seed {seed}')
        # Inicializar mezcla para K-means
        mixture, post = common.init(X, K, seed)
        #print(f'mixture = {mixture} and post = {post.ndim}')

        # Ejecutar K-means
        mixture, post, log_likehood = naive_em.run(X, mixture, post)
        #print(f'for kmeans, mixture = {mixture}, post = {post.ndim}, cost = {cost}')

        # Verificar si es el mejor costo encontrado hasta ahora para este K
        if log_likehood > highest_likehood:
            highest_likehood = log_likehood
            best_mix = mixture

    # Guardar el mejor costo y mezcla
    best_log_likehood[K] = highest_likehood
    best_mixtures[K] = best_mix

    # Graficar la mejor solución para este K
    common.plot(X, best_mix, post, f"K={K}")

# Imprimir resultados
for K in Ks:
    print(f"Cost|K={K} = {best_log_likehood[K]}")
'''
# Ks = [1, 2, 3, 4]
# seeds = [0, 1, 2, 3, 4]

# best_log_likelihood = {}
# best_mixtures = {}
# best_posts = {}

# for K in Ks:
#     print(f'Starting K = {K}')
#     highest_likelihood = float('-inf')
#     best_mix = None
#     best_post = None

#     for seed in seeds:
#         print(f'Starting seed {seed}')
#         mixture, post = common.init(X, K, seed)
#         mixture, post, log_likelihood = naive_em.run(X, mixture, post)

#         if log_likelihood > highest_likelihood:
#             highest_likelihood = log_likelihood
#             best_mix = mixture
#             best_post = post

#     best_log_likelihood[K] = highest_likelihood
#     best_mixtures[K] = best_mix
#     best_posts[K] = best_post

# # Crear títulos para los subplots
# titles_dict = {K: f"K = {K}" for K in Ks}

# # Dibujar todos los gráficos al mismo tiempo
# common.plot_multi(X, best_mixtures, best_posts, titles_dict)

# # Imprimir resultados
# for K in Ks:
#     print(f"Log-likelihood | K={K} = {best_log_likelihood[K]}")
'''
Ks = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]

best_cost = {}
best_mixtures = {}
best_posts = {}

for K in Ks:
    print(f'Starting K = {K}')
    lowest_cost = float('inf')
    best_mix = None
    best_post = None

    for seed in seeds:
        print(f'Starting seed {seed}')
        mixture, post = common.init(X, K, seed)
        mixture, post, cost = kmeans.run(X, mixture, post)

        if cost < lowest_cost:
            lowest_cost = cost
            best_mix = mixture
            best_post = post

    best_cost[K] = lowest_cost
    best_mixtures[K] = best_mix
    best_posts[K] = best_post

# Crear títulos
titles_dict = {K: f"K = {K}" for K in Ks}

# Graficar todos juntos
common.plot_multi(X, best_mixtures, best_posts, titles_dict)

# Imprimir resultados
for K in Ks:
    print(f"Cost | K={K} = {best_cost[K]}")
'''
### PREGUNTA 4 ENCONTRAR EL MEJOR K  y BCI

Ks = [1,2,3,4]
highest_bic= float('-inf')

for K in Ks:
    print(f'Starting K = {K}')

    # Inicializar mezcla para K-means
    mixture, post = common.init(X, K, 0)
    #print(f'mixture = {mixture} and post = {post.ndim}')

    # Ejecutar K-means
    mixture, post, log_likehood = naive_em.run(X, mixture, post)

    bic_value = common.bic(X, mixture, log_likehood)
    # Verificar si es el mejor costo encontrado hasta ahora para este K
    if bic_value > highest_bic:
        highest_bic = bic_value
        best_k = K

print(f'Best K = {best_k}')
print(f'Best bic = {highest_bic}')

