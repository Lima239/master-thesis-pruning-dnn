import torch
import numpy as np
from sklearn.cluster import SpectralClustering
from collections import defaultdict
from scipy.spatial.distance import cdist
import numpy as np

def build_mutual_knn_graph(X, n_neighbors):
    cosine_matrix = cdist(X, X, metric='cosine')
    cosine_matrix = abs(1 - cosine_matrix)
    np.fill_diagonal(cosine_matrix, 0)

    n = X.shape[0]
    knn_mask = np.zeros_like(cosine_matrix, dtype=bool)
    for i in range(n):
        top_k = np.argsort(cosine_matrix[i])[-n_neighbors:]
        knn_mask[i, top_k] = True

    mutual_knn = np.logical_and(knn_mask, knn_mask.T)
    epsilon = 1e-3
    affinity = cosine_matrix * mutual_knn.astype(float) + epsilon
    np.fill_diagonal(affinity, 0)
    return affinity

"""
Build k balanced clusters of size `cluster_size` using cosine similarity.
Underfilled clusters are filled by picking unused points that are closest to their centers.
"""
def balance_clusters(X, labels, k, cluster_size):
    n = X.shape[0]
    all_indices = set(range(n))

    clusters_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters_dict[label].append(idx)

    cluster_selections = [[] for _ in range(k)]
    cluster_centers = [None] * k
    used_indices = set()

    for c in range(k):
        indices = clusters_dict[c]
        cluster_points = X[indices]
        center = cluster_points.mean(axis=0, keepdims=True)
        cluster_centers[c] = center

        cosine_dist = cdist(cluster_points, center, metric="cosine")
        cosine_sim = np.abs(1 - cosine_dist).flatten()

        top_k = min(cluster_size, len(indices))
        top_indices = np.argsort(cosine_sim)[-top_k:]
        selected = [indices[i] for i in top_indices]

        cluster_selections[c].extend(selected)
        used_indices.update(selected)

    unused_indices = list(all_indices - used_indices)
    unused_points = X[unused_indices]

    for c in range(k):
        need = cluster_size - len(cluster_selections[c])
        if need > 0:
            center = cluster_centers[c]
            print("center", center)
            dist = cdist(unused_points, center, metric="cosine")
            sim = np.abs(1 - dist).flatten()

            best_indices = np.argsort(sim)[-need:]  # take most similar
            chosen = [unused_indices[i] for i in best_indices]

            cluster_selections[c].extend(chosen)
            used_indices.update(chosen)

            for idx in sorted(best_indices, reverse=True):
                del unused_indices[idx]
                unused_points = np.delete(unused_points, idx, axis=0)

            print(f"Cluster {c} filled with closest: {chosen}")

        assert len(cluster_selections[c]) == cluster_size, f"Cluster {c} still underfilled!"

    balanced_indices = [i for cluster in cluster_selections for i in cluster]
    return balanced_indices

def cluster_with_balanced_spectral_KNN(X, k):
    X = X.detach().cpu().numpy()
    #X_norm = normalize(X_np, norm='l2')
    n = X.shape[0]
    cluster_size = n // k

    affinity = build_mutual_knn_graph(X, 2 * cluster_size)

    spectral = SpectralClustering(
        n_clusters=k,
        affinity='precomputed',
        assign_labels='discretize',
        random_state=42
    )
    labels = spectral.fit_predict(affinity)

    balanced_indices = balance_clusters(X, labels, k, cluster_size)

    return torch.tensor(balanced_indices)


def compute_clustering_permutations_SPECTRAL_KNN_balanced(X, k1, k2):
    row_clusters_perm = cluster_with_balanced_spectral_KNN(X, k1)
    column_clusters_perm = cluster_with_balanced_spectral_KNN(X.T, k2)
    return row_clusters_perm, column_clusters_perm


if __name__ == "__main__":
    # X = torch.randn(64, 64)
    # rows, cols = compute_clustering_permutations_SPECTRAL(X, k=4)
    # print("Row permutation:", rows)
    # print("Column permutation:", cols)
    # X = torch.tensor([
    #     [10, 20, 30, 40, 50, 60, 70, 80, 90],  # Outlier
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Cluster 4 (identical to row 7)
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Cluster 1
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Cluster 1 (identical to row 1)
    #     [2, 4, 6, 8, 10, 12, 14, 16, 18],  # Cluster 2
    #     [2, 4, 6, 8, 10, 12, 14, 16, 18],  # Cluster 2 (identical to row 3)
    #     [3, 6, 9, 12, 15, 18, 21, 24, 27],  # Cluster 3
    #     [3, 6, 9, 12, 15, 18, 21, 24, 27],  # Cluster 3 (identical to row 5)
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Cluster 4 (constant row)
    # ])
    #
    # row_clusters_perm, column_clusters_perm = compute_clustering_permutations_SPECTRAL_KNN_balanced(X, 3, 3)
    # print(row_clusters_perm)
    # print(column_clusters_perm)
    X = torch.load("/inputs/fc1.pt")
    print(X.shape)

    P_rows, P_columns = compute_clustering_permutations_SPECTRAL_KNN_balanced(X, 4, 4)
    X = X[P_rows, :]
    X = X[:, P_columns]
