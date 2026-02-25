# HUMLPY: High-dimensional Undirected Mixed graph Learning in PYthon

This site contains the documentation for the `humlpy` Python package.

## Overview

**humlpy** implements the methodology for learning sparse undirected graphical models
from arbitrary mixed data (any combination of continuous and ordinal variables)
presented in:

> Göbler, K., Drton, M., Mukherjee, S. and Miloschewski, A. (2024).
> **High-dimensional undirected graphical models for arbitrary mixed data.**
> *Electronic Journal of Statistics*, 18(1), 2339–2404.
> <https://doi.org/10.1214/24-EJS2254>

Given a (high-dimensional) dataset with continuous and/or ordinal variables the
package estimates the latent correlation matrix with pair-type-specific estimators
and recovers the graph structure via graphical lasso with eBIC model selection.

## Quick Start

Generate data from a latent sparse Gaussian model and binarise half the
columns, then recover the graph.

```python
import numpy as np
import pandas as pd
from scipy import stats
from humlpy import MixedGraphicalLasso

rng = np.random.default_rng(0)
n, d = 400, 20

# --- Sparse precision matrix ------------------------------------------------
# Start from the identity, add signal to ~12 randomly chosen pairs,
# then enforce positive-definiteness via diagonal dominance.
precision = np.eye(d)
pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
true_edges = set()
for idx in rng.choice(len(pairs), size=12, replace=False):
    i, j = pairs[idx]
    precision[i, j] = precision[j, i] = 0.5
    true_edges.add(frozenset((f"x{i}", f"x{j}")))
np.fill_diagonal(precision, np.abs(precision).sum(axis=1) + 0.1)

# --- Latent multivariate nonparanormal data ----------------------------------------
cov = np.linalg.inv(precision)
X = rng.multivariate_normal(np.zeros(d), cov, size=n)
X = np.sign(X) * np.power(np.abs(X), 1.5)  # nonparanormal transform

# --- Binarise first half of columns (probit / quantile transform) -----------
n_bin = d // 2
p_bin = rng.uniform(0.4, 0.6, size=n_bin)
data = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
for i in range(n_bin):
    u = stats.norm.cdf(stats.zscore(X[:, i]))
    data.iloc[:, i] = stats.binom.ppf(u, n=1, p=p_bin[i])

# --- Fit the mixed graphical model ------------------------------------------
mgl = MixedGraphicalLasso().fit(data)

print(f"Selected alpha:  {mgl.alpha_:.4f}")
print(f"Number of edges: {mgl.n_edges_}")
print(f"Edges:           {mgl.graph_.edges}")

# --- Evaluation -------------------------------------------------------------
recovered = {frozenset(e) for e in mgl.graph_.edges}
tp = len(true_edges & recovered)
tpr = tp / len(true_edges)
fpr = (len(recovered) - tp) / (d * (d - 1) // 2 - len(true_edges))
print(f"TPR: {tpr:.2f}  FPR: {fpr:.2f}")
```

## Installation

```bash
pip install -e .
```

## Main API

| Class / Function      | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `MixedGraphicalLasso` | Full pipeline: correlation → glasso path → eBIC selection → `UGRAPH` |
| `SampleCorrelation`   | Latent correlation matrix for mixed data                             |
| `omega_select`        | eBIC model selection over a glasso regularisation path               |
| `UGRAPH`              | Undirected graph class                                               |

## References

- Göbler, K., Drton, M., Mukherjee, S. and Miloschewski, A. (2024).
  High-dimensional undirected graphical models for arbitrary mixed data.
  *Electronic Journal of Statistics*, 18(1), 2339–2404.
  <https://doi.org/10.1214/24-EJS2254>

- Foygel, R. and Drton, M. (2010).
  Extended Bayesian Information Criteria for Gaussian Graphical Models.
  *Advances in Neural Information Processing Systems*, 23, 604–612.

- Liu, H., Lafferty, J. and Wasserman, L. (2009).
  The nonparanormal: Semi-parametric estimation of high dimensional undirected graphs.
  *Journal of Machine Learning Research*, 10(80), 2295–2328.

See the [API Reference](reference.md) for detailed documentation.
