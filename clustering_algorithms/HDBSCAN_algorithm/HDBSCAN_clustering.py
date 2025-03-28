import numpy as np
import torch
import os
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import SpectralClustering
import hdbscan

def cluster_with_hdbscan_high_accuracy(X):
    X_np = X.detach().cpu().numpy().astype(np.float64)
    cosine_dist = cosine_distances(X_np).astype(np.float64)

    clusterer = hdbscan.HDBSCAN(
        metric='precomputed',
        min_cluster_size=10,
        min_samples=10,
        cluster_selection_method='eom',
        alpha=0.5,
        allow_single_cluster=True,
        prediction_data=True
    )
    labels = clusterer.fit_predict(cosine_dist)

    unique_labels = sorted(set(labels) - {-1})
    clusters = [[] for _ in unique_labels]
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    for idx, lbl in enumerate(labels):
        if lbl != -1:
            clusters[label_to_index[lbl]].append(idx)

    return torch.tensor([i for group in clusters for i in group], dtype=torch.long)

def compute_clustering_permutations_HDBSCAN(X):
    row_clusters_perm = cluster_with_hdbscan_high_accuracy(X)
    column_clusters_perm = cluster_with_hdbscan_high_accuracy(X.T)

    return row_clusters_perm, column_clusters_perm

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #      print("Usage: python3 cosine_dist_mip.py <input_file_path>")
    # else:
    input_path = "/Users/KlaudiaLichmanova/Desktop/school5/Diplomovka/codebase/master/master-thesis-pruning-dnn/inputs/real_weights/pytorch_weights_36x36_1.pt"
    X = torch.load(input_path)
    #
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

