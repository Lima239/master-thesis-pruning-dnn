from scipy.spatial.distance import cdist
import numpy as np
import torch

def greedy_cosine_clusters(X, k, refine_iters=30):
    X_np = X.detach().cpu().numpy()
    n = X_np.shape[0]
    assert n % k == 0, "n must be divisible by k"
    cluster_size = n // k

    sim_matrix = 1 - cdist(X_np, X_np, metric='cosine')
    np.fill_diagonal(sim_matrix, -np.inf)

    assigned = set()
    clusters = [[] for _ in range(k)]

    for c in range(k):
        remaining = list(set(range(n)) - assigned)
        seed_scores = sim_matrix[remaining][:, remaining].sum(axis=1)
        seed = remaining[np.argmax(seed_scores)]

        neighbors = np.argsort(sim_matrix[seed])[::-1]
        chosen = []
        for idx in neighbors:
            if idx not in assigned:
                chosen.append(idx)
            if len(chosen) == cluster_size:
                break

        clusters[c] = chosen
        assigned.update(chosen)

    for _ in range(refine_iters):
        for i in range(n):
            current_cluster = next(ci for ci, c in enumerate(clusters) if i in c)
            best_gain = 0
            best_cluster = current_cluster
            for target_cluster in range(k):
                if target_cluster == current_cluster or len(clusters[target_cluster]) >= cluster_size:
                    continue
                gain = (
                    sum(sim_matrix[i][j] for j in clusters[target_cluster]) -
                    sum(sim_matrix[i][j] for j in clusters[current_cluster] if j != i)
                )
                if gain > best_gain:
                    best_gain = gain
                    best_cluster = target_cluster

            if best_cluster != current_cluster:
                clusters[current_cluster].remove(i)
                clusters[best_cluster].append(i)

    perm = [i for cluster in clusters for i in cluster]
    return torch.tensor(perm)

def compute_clustering_permutations_COS_GREEDY_LOCAL(X, k1, k2):
    row_perm = greedy_cosine_clusters(X, k1)
    col_perm = greedy_cosine_clusters(X.T, k2)
    return row_perm, col_perm