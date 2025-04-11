import torch
import numpy as np
from scipy.spatial.distance import cdist
import random
import math

def cosine_similarity_matrix(X):
    X_np = X.detach().cpu().numpy()
    sim = 1 - cdist(X_np, X_np, metric="cosine")
    np.fill_diagonal(sim, 0)  #zero self similarity
    return sim

def score_clusters(sim_matrix, clusters):
    score = 0.0
    for cluster in clusters:
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                score += sim_matrix[cluster[i], cluster[j]]
    return score

def initialize_clusters(n, k):
    indices = np.random.permutation(n)
    cluster_size = n // k
    return [list(indices[i*cluster_size:(i+1)*cluster_size]) for i in range(k)]

#simulated annealing optimizer for balanced k-clustering maximizing cosine similarity
def simulated_annealing(X, k, steps= 1000, T_start=1.0, T_end=0.01):
    X_np = X.detach().cpu().numpy()
    n = X_np.shape[0]
    assert n % k == 0, "X must be evenly divisible by k"
    cluster_size = n // k

    sim_matrix = cosine_similarity_matrix(X)

    clusters = initialize_clusters(n, k)
    cluster_map = np.zeros(n, dtype=int)
    for i, c in enumerate(clusters):
        for idx in c:
            cluster_map[idx] = i

    current_score = score_clusters(sim_matrix, clusters)
    best_score = current_score
    best_clusters = [c.copy() for c in clusters]

    for step in range(steps):
        T = T_start * ((T_end / T_start) ** (step / steps))

        #pick two clusters
        c1, c2 = random.sample(range(k), 2)
        if not clusters[c1] or not clusters[c2]:
            continue

        #pick one entry from each
        i1 = random.choice(clusters[c1])
        i2 = random.choice(clusters[c2])

        #swap
        clusters[c1].remove(i1)
        clusters[c2].remove(i2)
        clusters[c1].append(i2)
        clusters[c2].append(i1)

        new_score = score_clusters(sim_matrix, clusters)
        delta = new_score - current_score

        if delta > 0 or math.exp(delta / T) > random.random():
            current_score = new_score
            cluster_map[i1], cluster_map[i2] = c2, c1
            if new_score > best_score:
                best_score = new_score
                best_clusters = [c.copy() for c in clusters]
        else:
            #undo swap
            clusters[c1].remove(i2)
            clusters[c2].remove(i1)
            clusters[c1].append(i1)
            clusters[c2].append(i2)

    permuted_indices = [i for cluster in best_clusters for i in cluster]
    return torch.tensor(permuted_indices, dtype=torch.long)


def compute_clustering_permutations_ANNEAL(X, k1, k2, steps=1000):
    row_perm = simulated_annealing(X, k1, steps)
    col_perm = simulated_annealing(X.T, k2, steps)
    return row_perm, col_perm

