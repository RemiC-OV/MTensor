# MTensor Toolbox

This is the public repository for the m-tensor toolbox and proposed examples. It comes as is with no guarantees, some algorithms are not yet tested and require more theoretical work to demonstrate porperly. The provided toolbox is adapted for sparse cores as well as for dense ones. The sparse version might not be as well kept up to date as the dense version.

If you use this work, please cite the [preprint](https://hal.science/hal-05500936) as

>Rémi Cloarec, Sebastian Rodriguez, Xavier Kestelyn, Francisco Chinesta. The M-Tensor Format: Optimality in High Dimensional Regression for Nonlinear Models with Scarce Data. 2026. ⟨hal-05500936⟩

It is also accessible [on ArXiv](https://arxiv.org/abs/2602.08509). We presented a conference [poster](https://hal.science/hal-05546159) 

>Rémi Cloarec, Sebastian Rodriguez, Lila Achour, Xavier Kestelyn, Francisco Chinesta. A New Low Rank Tensor Format for High Dimensional Sparse Regression. Mortech 2025, 7th International Workshop on Model Order Reduction Techniques, Nov 2025, Zaragoza (Universidad de Zaragoza), Spain. ⟨hal-05546159⟩

## Examples
The toolbow comes with a few examples to demonstrate the good scaling of m-tensors with dimension and its robustness. Low dimensional examples are used for comprehension but some do not respect some assumptions and therefore sometimes produce errors (negative distance in ALID), this is solved by providing larger 1D bases (piecewise linear or locally supported RBF).

### Rosenbrock benchmark

We propose this benchmark out of its usual optimization context because it allows us to easily scale in dimensions and therefore demonstrate complexity scaling.

### Lorenz attractor

The Lorenz attractor is a usual benchmark in dynamical systems that allows to demonstrate the quality of an approximation if it holds the right solution over large time windows.

### Kuramoto oscillators

Kuramoto oscilators are also a dynamical system but it scales with the number of scillators, therefore it allows us to demonstrate complexity scaling.

### Polynomial toy problem

The polynomial toy problem is excessively simple and demonstrates the exact correspondance of our framework with matrix operations on a case that may be understood very easily.

## References

Our references are all cited in the [preprint](https://hal.science/hal-05500936) but a few of the main ones, noticeably in terms of kernel interpretation and properties are

> Peter J. Baddoo et al. “Kernel learning for robust dynamic mode decomposition:
linear and nonlinear disambiguation optimization”. In: Proceedings of the Royal
Society A: Mathematical, Physical and Engineering Sciences 478.2260 (2022),
p. 20210830. doi: 10.1098/rspa.2021.0830

>Y. Engel, S. Mannor, and R. Meir. “The kernel recursive least-squares algo
rithm”. In: IEEE Transactions on Signal Processing 52.8 (2004), pp. 2275–2285.
doi: 10.1109/TSP.2004.830985.

> Stefan Klus and Patrick Gelß. “Tensor-Based Algorithms for Image Classifica
tion”. In: Algorithms 12.11 (2019). issn: 1999-4893. doi: 10.3390/a12110240.