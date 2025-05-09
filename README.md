# Master Thesis: Improving Structured Pruning of Deep Neural Networks

Master's thesis focused on compressing neural networks using structured pruning. It includes all the code and experiments for applying **Monarch decomposition** and improving it with **permutation algorithms**.

We test two approaches:
- **ILP-based permutation**
- **Spectral KNN permutation**

## How to Use

To run a clustering algorithm (like ILP or Spectral KNN), go to:
- `clustering_algorithms/run.sh`

To compute error of Monarch decomposition, go to:
- `monarch_decomposition/testMonarchDecomposition.py`


## Jupyter Notebooks

There are two notebooks for experimenting on real models:

- `monarch_permutations_activations.ipynb`  
Applies Monarch + permutation (ILP or KNN), using activation statistics to improve clustering.

- `only_monarch_activations.ipynb`  
Applies only Monarch decomposition, using activation stats but no permutation.
