import numpy as np
import torch
import os
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import SpectralClustering
import hdbscan
from collections import defaultdict
from scipy.spatial.distance import cdist
def cluster_with_hdbscan_high_accuracy(X, k):
    X_np = X.detach().cpu().numpy().astype(np.float64)

    cosine_matrix = cdist(X, X, metric='cosine')
    cosine_matrix = abs(1 - cosine_matrix)
    print(cosine_matrix)
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=k,
        min_samples=2,
        cluster_selection_method='eom',
        alpha=0.05,
        allow_single_cluster=True,
        prediction_data=True
    )

    labels = clusterer.fit_predict(cosine_matrix)

    # Group indices by cluster label
    raw_clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        if lbl != -1:
            raw_clusters[lbl].append(idx)

    sorted_indices = [i for cluster in raw_clusters.values() for i in cluster]

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

def compute_clustering_permutations_HDBSCAN(X, k):
    row_clusters_perm = cluster_with_hdbscan_high_accuracy(X, k)
    column_clusters_perm = cluster_with_hdbscan_high_accuracy(X.T, k)

    return row_clusters_perm, column_clusters_perm

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #      print("Usage: python3 cosine_dist_mip.py <input_file_path>")
    # else:
    torch.set_printoptions(linewidth=2000)
    input_path = "/Users/KlaudiaLichmanova/Desktop/school5/Diplomovka/codebase/master/master-thesis-pruning-dnn/inputs/real_weights/pytorch_weights_36x36_1.pt"
    X = torch.load(input_path)

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

    X = compute_clustering_permutations_HDBSCAN(X, 3)
    print(X)
    # k = int(np.sqrt(X.shape[0]))
    # X, row_clusters = hdbscan_clustering(X, k)
    #
    # X, column_clusters = hdbscan_clustering(X.T, k)
    #
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # output_dir = os.path.join(script_dir, "output")
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # file_name = os.path.splitext(os.path.basename(input_path))[0]
    # save_name = f"{file_name}_cosine_abs_spectral.pt"
    #
    # save_path = os.path.join(output_dir, save_name)
    # torch.save(X, save_path)

