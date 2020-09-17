# Spin Hamiltonian

Code in Python (with numpy and scipy) for calculation of a Spin Hamiltonian in the context of STM experiments.

For examples how to use the code look at the Jupyter Notebook "FeOnNDE.ipynd" and "FeOnNStevens.ipynb". Those notebooks calculate the differential conductance for an iron atom bound on a nitrogen site of a copper nitrite substrate with anisotropy operators and Stevens operators, respectively. 

## Features
* Anisotropy operators (D and E) or Stevens operators models
* Single-atom or chain/loop of multiple atoms
* Sparse matrix calculation when needed
* Cached Kronecker space calculation
* On-disk cached results for long calculations
* Multiple algorithms for solving the rates equations
* Heavily relied on numpy and scipy for the computation

## Dependencies
* Numpy
* Scipy
* tqdm


## Publication
Rejali, R., Coffey, D., Gobeil, J. et al. Complete reversal of the atomic unquenched orbital moment by a single electron. npj Quantum Mater. 5, 60 (2020). https://doi.org/10.1038/s41535-020-00262-w