import torch

"""    
This function projects input matrix onto two block diagonal matrices L and R using SVD.
"""
def project_on_monarch_matrices(M):
  n = M.size(0)
  m = int(n ** 0.5)
  assert m * m == n, "n must be a perfect square"

  # ensure the input is in float32 if it's in half precision
  # if B.dtype == torch.float16:
  #     B = B.float()

  # reshape B into a 4D tensor A
  A = M.view(m, m, m, m).permute(0, 2, 1, 3)

  L_blocks = []
  R_blocks = []

  for j in range(m):
    columns_U = []
    rows_VT = []
    for k in range(m):
      M_jk = A[:, j, k, :]  # size m x m

      # SVD of M_jk
      U, S, VT = torch.linalg.svd(torch.tensor(M_jk).float(), full_matrices=False)
      # get the first column of U
      u_jk = U[:, 0]
      # first row of V^T
      v_jk = VT[0, :]

      columns_U.append(u_jk * S[0])
      rows_VT.append(v_jk.unsqueeze(0))

    L_blocks.append(torch.stack(columns_U, dim=1))
    R_blocks.append(torch.cat(rows_VT, dim=0))

  # convert block lists into block-diagonal matrices
  L = torch.block_diag(*L_blocks)
  R = torch.block_diag(*R_blocks)

  return L, R

"""
Creates permutation matrix. The matrix P reorders the rows and columns of the input block-diagonal matrix that blocks 
along the diagonal are shifted to form a new matrix where blocks are arranged row-wise across the matrix
"""
def monarch_permutation_matrix(m):
  m = int(m)
  n = m * m

  P = torch.zeros((n, n), dtype=torch.float32)

  for i in range(n):
    P[i, (i % m) * m + i // m] = 1

  return P


"""
Returns the reconstructed Monarch matrix, calculated as M = PLP^T R.
"""
def reconstruct_monarch_matrix(L, R, m):
  m = int(m)

  P = monarch_permutation_matrix(m)

  PL = torch.matmul(P, L)
  PTR = torch.matmul(torch.transpose(P, 0, 1), R)

  M = torch.matmul(PL, PTR)
  return M

"""
This function converts a block-structured matrix back into its original matrix form.
"""
def block_matrix_to_original(block_matrix):
  n = block_matrix.shape[0]
  block_size = int(n ** 0.5)
  original = torch.zeros((n, n))

  for i in range(n):
    for j in range(n):
      block_row = (i // block_size) * block_size + (j // block_size * block_size) // block_size
      block_col = (i % block_size) * block_size + j % block_size

      original[i][j] = block_matrix[block_row][block_col]

  return original