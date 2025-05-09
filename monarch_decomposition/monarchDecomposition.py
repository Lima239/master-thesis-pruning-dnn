import torch
import torch.nn as nn

def permute(nb, size, L):
    out = []
    perm_base = torch.arange(size)
    parts = perm_base.chunk(nb*nb)
    for i in range(nb):
      out += parts[i::nb]

    orig_shape = L.shape
    L = L.flatten(0, -2)

    out = L.T[torch.cat(out)].T
    return out.reshape(orig_shape)

def monarch_decomposition(W, n_blocks=4, rank=None):
    s0, s1 = W.shape[0] // n_blocks, W.shape[1] // n_blocks

    if rank is None:
        bs = W.shape[0] // n_blocks, W.shape[1] // n_blocks
        rank = int((bs[0] * bs[1]) / (bs[0] + bs[1]) / 2)  # around 50%

    print("rank ", rank)
    mid = rank * n_blocks

    L = nn.Parameter(torch.zeros((n_blocks, W.shape[1]//n_blocks, mid), device=W.device))
    R = nn.Parameter(torch.zeros((n_blocks, mid, W.shape[0]//n_blocks), device=W.device))

    for i in range(n_blocks):
        for j in range(n_blocks):
            part = W[i*s0:i*s0+s0, j*s1:j*s1+s1]

            U, s, Vh = torch.linalg.svd(part, full_matrices=False)
            s = s[:rank]
            U = U[:, :rank] * s.sqrt()
            Vh = Vh[:rank] * s.sqrt().unsqueeze(1)

            L.data[j, :, rank*i:rank*i+rank] = (Vh).T
            R.data[i, rank*j:rank*j+rank] = U.T


    L = [L[i] for i in range(L.shape[0])]
    L = torch.block_diag(*L)

    R = [R[i] for i in range(R.shape[0])]
    R = torch.block_diag(*R)

    L = permute(n_blocks, mid*n_blocks, L)
    W_approx = L @ R

    return W_approx.T

def compute_reconstruction_error(W: torch.Tensor, W_approx: torch.Tensor):
  error = torch.norm(W - W_approx, p='fro') ** 2
  base = torch.norm(W, p='fro') ** 2
  return (error / base).item()