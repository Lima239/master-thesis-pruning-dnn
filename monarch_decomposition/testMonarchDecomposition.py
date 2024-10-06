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

    M_transformed = monarchDecomposition.block_matrix_to_original(M_reconstructed)

    print('\nReconstructed 2 matrix from monarch decomposition:')
    print(M_transformed)


if __name__ == "__main__":
    input_matrix_path = 'inputs/test_matrices/9x9.pt'
    M = torch.load(input_matrix_path)
    monarch_decomposition_analysis(M)
