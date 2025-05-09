import torch
from monarchDecomposition import monarch_decomposition, compute_reconstruction_error


def monarch_decomposition_error(M):
    M_approx = monarch_decomposition(M)
    error = compute_reconstruction_error(M, M_approx)
    print('\nError of the matrix is:')
    print(error)

if __name__ == "__main__":
    torch.set_printoptions(linewidth=500)

    M = torch.load("../inputs/fc1.pt")

    monarch_decomposition_error(M)
