import torch
import numpy as np
from scipy.spatial.distance import cdist
import random
import math
from tqdm import trange


def cosine_similarity_matrix(X):
    X_np = X.detach().cpu().numpy()
    sim = 1 - cdist(X_np, X_np, metric="cosine")
    np.fill_diagonal(sim, 0)
    return sim


def initialize_balanced_clusters(n, k):
    indices = np.random.permutation(n)
    cluster_size = n // k
    return [list(indices[i*cluster_size:(i+1)*cluster_size]) for i in range(k)]


def normalized_score(sim_matrix, clusters, penalty_threshold=0.9):
    score = 0.0
    inter_cluster_penalty = 0.0
    cluster_map = {}
    for cluster_idx, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_map[idx] = cluster_idx

    for cluster in clusters:
        if len(cluster) <= 1:
            continue
        sims = sim_matrix[np.ix_(cluster, cluster)]
        score += np.sum(sims) / (len(cluster) * (len(cluster) - 1))

    # Add penalty for splitting highly similar points across clusters
    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1, sim_matrix.shape[1]):
            if sim_matrix[i, j] > penalty_threshold and cluster_map[i] != cluster_map[j]:
                inter_cluster_penalty += sim_matrix[i, j]

    return score - 0.1 * inter_cluster_penalty  # scale penalty weight


def simulated_annealing_balanced_clustering(X, k, steps=3000, T_start=2.0, T_end=0.05,
                                            swap_size=2, penalty_threshold=0.9):
    X_np = X.detach().cpu().numpy()
    n = X_np.shape[0]
    assert n % k == 0, "Must be evenly divisible"

    sim_matrix = cosine_similarity_matrix(X)
    cluster_size = n // k

    clusters = initialize_balanced_clusters(n, k)
    current_score = normalized_score(sim_matrix, clusters, penalty_threshold)
    best_clusters = [c.copy() for c in clusters]
    best_score = current_score

    for step in trange(steps, desc="Annealing"):
        T = T_start - (T_start - T_end) * (step / steps)  # linear cooling

        # Pick two clusters
        c1, c2 = random.sample(range(k), 2)

        if len(clusters[c1]) < swap_size or len(clusters[c2]) < swap_size:
            continue

        # Pick elements
        i1 = random.sample(clusters[c1], swap_size)
        i2 = random.sample(clusters[c2], swap_size)

        # Swap them
        for a, b in zip(i1, i2):
            clusters[c1].remove(a)
            clusters[c2].remove(b)
            clusters[c1].append(b)
            clusters[c2].append(a)

        new_score = normalized_score(sim_matrix, clusters, penalty_threshold)
        delta = new_score - current_score

        if delta > 0 or math.exp(delta / T) > random.random():
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_clusters = [c.copy() for c in clusters]
        else:
            # Undo swap
            for a, b in zip(i1, i2):
                clusters[c1].remove(b)
                clusters[c2].remove(a)
                clusters[c1].append(a)
                clusters[c2].append(b)

    return torch.tensor([i for cluster in best_clusters for i in cluster], dtype=torch.long)


def compute_clustering_permutations_ANNEAL_PRO(X, Y, k1, k2, steps=5000, n_retries=5):
    best_score = float("-inf")
    best_row_perm, best_col_perm = None, None

    for run in range(n_retries):
        row_perm = simulated_annealing_balanced_clustering(X, k1, steps=steps)
        col_perm = simulated_annealing_balanced_clustering(Y.T, k2, steps=steps)

        # Score the result
        sim = cosine_similarity_matrix(X)
        clusters = [row_perm[i::k1].tolist() for i in range(k1)]
        score = normalized_score(sim, clusters)

        if score > best_score:
            best_score = score
            best_row_perm = row_perm
            best_col_perm = col_perm

    return best_row_perm, best_col_perm
