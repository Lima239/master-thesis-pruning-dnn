import torch
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
from collections import defaultdict
import torch.nn.functional as F

def cluster_with_spectral_cosine(X: torch.Tensor, k: int) -> torch.Tensor:
    """
    Spectral clustering using cosine similarity, returning exactly k equal-size clusters.
    Returns a flattened torch.Tensor of ordered indices like the HDBSCAN-based version.
    """
    X_np = F.normalize(X.float(), p=2, dim=1)
    X_np = X_np.detach().cpu().numpy()
    cosine_sim = cdist(X_np, X_np, metric='cosine')
    cosine_sim = 1 - cosine_sim
    np.fill_diagonal(cosine_sim, 1.0)
    cosine_sim = np.clip(cosine_sim, 0, 1)

    spectral = SpectralClustering(
        n_clusters=k,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=0
    )
    labels = spectral.fit_predict(cosine_sim)

    clusters_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters_dict[label].append(idx)

    sorted_indices = [i for cluster in clusters_dict.values() for i in cluster]

    all_indices = set(range(X.shape[0]))
    unassigned = list(all_indices - set(sorted_indices))
    sorted_indices.extend(unassigned)

    n = X.shape[0]
    if n % k != 0:
        raise ValueError(f"Invalid number of clusters: {k}. Cannot evenly divide {n} samples.")

    cluster_size = n // k
    clusters = [sorted_indices[i*cluster_size:(i+1)*cluster_size] for i in range(k)]

    print(clusters)
    return torch.tensor(clusters).flatten()


def compute_clustering_permutations_SPECTRAL(X: torch.Tensor, k1: int, k2: int):
    row_clusters_perm = cluster_with_spectral_cosine(X, k1)
    column_clusters_perm = cluster_with_spectral_cosine(X.T, k2)
    return row_clusters_perm, column_clusters_perm


if __name__ == "__main__":
    # X = torch.randn(64, 64)
    # rows, cols = compute_clustering_permutations_SPECTRAL(X, k=4)
    # print("Row permutation:", rows)
    # print("Column permutation:", cols)
    X = torch.tensor([
        [10, 20, 30, 40, 50, 60, 70, 80, 90],  # Outlier
        [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Cluster 4 (identical to row 7)
        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Cluster 1
        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Cluster 1 (identical to row 1)
        [2, 4, 6, 8, 10, 12, 14, 16, 18],  # Cluster 2
        [2, 4, 6, 8, 10, 12, 14, 16, 18],  # Cluster 2 (identical to row 3)
        # [3, 6, 9, 12, 15, 18, 21, 24, 27],  # Cluster 3
        # [3, 6, 9, 12, 15, 18, 21, 24, 27],  # Cluster 3 (identical to row 5)
        # [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Cluster 4 (constant row)
    ])

    X = compute_clustering_permutations_SPECTRAL(X, 3, 3)
    print(X)