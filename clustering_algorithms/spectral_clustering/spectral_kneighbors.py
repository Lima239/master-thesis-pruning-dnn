import torch
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
from collections import defaultdict
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

def cluster_with_spectral_kneighbors(X, k):
    #X_np = F.normalize(X.float(), p=2, dim=1)
    X_np = X.detach().cpu().numpy()
    X_norm = normalize(X_np, norm='l2')
    #pca = PCA(n_components=min(32, X_np.shape[1]))
    #X_np_reduced = pca.fit_transform(X_np)
    n = X.shape[0]
    cluster_size = n // k

    affinity = kneighbors_graph(
        X_norm,
        n_neighbors=2*cluster_size,
        metric='cosine',
        include_self=False,
        mode='connectivity',  # or distance?
        n_jobs=-1
    )

    spectral = SpectralClustering(
        n_clusters=k,
        affinity='precomputed',
        assign_labels='discretize',
        n_init=10,
        random_state=42
    )
    labels = spectral.fit_predict(affinity)

    clusters_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters_dict[label].append(idx)

    sorted_indices = [i for cluster in clusters_dict.values() for i in cluster]

    all_indices = set(range(X.shape[0]))
    unassigned = list(all_indices - set(sorted_indices))
    sorted_indices.extend(unassigned)

    if n % k != 0:
        raise ValueError(f"Invalid number of clusters: {k}. Cannot evenly divide {n} samples.")


    clusters = [sorted_indices[i*cluster_size:(i+1)*cluster_size] for i in range(k)]

    print(clusters)
    return torch.tensor(clusters).flatten()


def compute_clustering_permutations_SPECTRAL_kneighbors(X, k1, k2):
    row_clusters_perm = cluster_with_spectral_kneighbors(X, k1)
    column_clusters_perm = cluster_with_spectral_kneighbors(X.T, k2)
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

    X = compute_clustering_permutations_SPECTRAL_kneighbors(X, 3, 3)
    print(X)