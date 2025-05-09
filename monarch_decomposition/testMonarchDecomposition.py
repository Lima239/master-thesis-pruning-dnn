import torch
import monarchDecomposition


def monarch_decomposition_analysis(M):
    print('Input matrix:')
    print(M)

    L, R = monarchDecomposition.project_on_monarch_matrices(M)

    print('\nBlock-diagonal matrix L:')
    print(L)
    print('\nBlock-diagonal matrix R:')
    print(R)

    m = int(M.shape[0] ** 0.5)
    M_reconstructed = monarchDecomposition.reconstruct_monarch_matrix(L, R, m)

    print('\nReconstructed 1 matrix from monarch decomposition:')
    print(M_reconstructed)

    #M_transformed = monarchDecomposition.block_matrix_to_original(M_reconstructed)
    M_transformed = M_reconstructed
    print('\nReconstructed 2 matrix from monarch decomposition:')
    print(M_transformed)

    error_matrix = M - M_transformed
    error = torch.norm(error_matrix, p='fro') ** 2

    print('\nError of the matrix is:')
    print(error)

if __name__ == "__main__":
    # input_matrix_path = '/Users/KlaudiaLichmanova/Desktop/school5/Diplomovka/codebase/master/master-thesis-pruning-dnn/inputs/real_weights/pytorch_weights_36x36_1.pt'
    #
    # M = torch.load(input_matrix_path)
    # print(M.size())
    # monarch_decomposition_analysis(M)
    torch.set_printoptions(linewidth=500)

    input_matrix_path = torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Cluster 1
            [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Cluster 1 (identical to row 1)
            [2, 4, 6, 8, 10, 12, 14, 16, 18],  # Cluster 2
            [2, 4, 6, 8, 10, 12, 14, 16, 18],  # Cluster 2 (identical to row 3)
            [3, 6, 9, 12, 15, 18, 21, 24, 27],  # Cluster 3
            [3, 6, 9, 12, 15, 18, 21, 24, 27],  # Cluster 3 (identical to row 5)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Cluster 4 (constant row)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # Cluster 4 (identical to row 7)
            [10, 20, 30, 40, 50, 60, 70, 80, 90],  # Outlier
        ])
    #M = torch.load(input_matrix_path)
    monarch_decomposition_analysis(input_matrix_path)
