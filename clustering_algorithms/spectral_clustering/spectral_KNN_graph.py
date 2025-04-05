import torch
import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
from collections import defaultdict
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_mutual_knn_graph(X_np, n_neighbors):
    sim_matrix = cosine_similarity(X_np)
    n = X_np.shape[0]
    knn_mask = np.zeros_like(sim_matrix, dtype=bool)

    # top-k per row
    for i in range(n):
        top_k = np.argsort(sim_matrix[i])[-n_neighbors:]
        knn_mask[i, top_k] = True

    mutual_knn = np.logical_and(knn_mask, knn_mask.T)
    affinity = sim_matrix * mutual_knn
    return affinity


def cluster_with_spectral_KNN(X, k):
    #X_np = F.normalize(X.float(), p=2, dim=1)
    X_np = X.detach().cpu().numpy()
    X_norm = normalize(X_np, norm='l2')
    #pca = PCA(n_components=min(32, X_np.shape[1]))
    #X_np_reduced = pca.fit_transform(X_np)
    n = X.shape[0]
    cluster_size = n // k

    affinity = build_mutual_knn_graph(X_np, 2 * cluster_size)

    spectral = SpectralClustering(
        n_clusters=k,
        affinity='precomputed',
        assign_labels='discretize',
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


def compute_clustering_permutations_SPECTRAL_KNN(X, k1, k2):
    row_clusters_perm = cluster_with_spectral_KNN(X, k1)
    column_clusters_perm = cluster_with_spectral_KNN(X.T, k2)
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

    X = compute_clustering_permutations_SPECTRAL_KNN(X, 3, 3)
    print(X)