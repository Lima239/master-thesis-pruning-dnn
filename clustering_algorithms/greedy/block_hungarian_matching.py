import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

#block-wise Hungarian matching
def blockwise_hungarian_permutation(X, k):
    X_np = X.detach().cpu().numpy()
    n = X_np.shape[0]
    assert n % k == 0, f"Cannot evenly divide {n} into {k} clusters."
    block_size = n // k

    # k-means to init row,column block groups
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_np)
    labels = kmeans.labels_

    final_perm = []

    for block_id in range(k):
        block_indices = np.where(labels == block_id)[0]
        if len(block_indices) < block_size:
            #fill from other blocks if this one is underfilled
            needed = block_size - len(block_indices)
            pool = list(set(range(n)) - set(block_indices))
            block_indices = np.concatenate([block_indices, pool[:needed]])
        elif len(block_indices) > block_size:
            block_indices = block_indices[:block_size]

        #cosine sim matrix within block
        block_vectors = X_np[block_indices]
        sim_matrix = 1 - cdist(block_vectors, block_vectors, metric="cosine")

        # hungarian matching, maximize total similarity
        cost_matrix = -sim_matrix  #because hungarian minimizes cost
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_block = block_indices[col_ind]
        final_perm.extend(matched_block)

    return torch.tensor(final_perm, dtype=torch.long)

def compute_clustering_permutations_HUNGARIAN(X, k1, k2):
    row_perm = blockwise_hungarian_permutation(X, k1)
    col_perm = blockwise_hungarian_permutation(X.T, k2)
    return row_perm, col_perm
