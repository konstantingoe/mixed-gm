# HUMPY: High-dimensional Undirected Mixed graph estimation in PYthon

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://konstantingoe.github.io/mixed-gm/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)](https://github.com/konstantingoe/mixed-gm/actions)

**humpy** (**H**igh-dimensional **U**ndirected **M**ixed graph estimation in **PY**thon) is a Python package for learning sparse undirected graphical models from
arbitrary mixed data — any combination of continuous and ordinal variables.
It is built upon the [hume R package](https://github.com/konstantingoe/hume) and implements the methodology developed in:

> Göbler, K., Drton, M., Mukherjee, S. and Miloschewski, A. (2024).
> **High-dimensional undirected graphical models for arbitrary mixed data.**
> *Electronic Journal of Statistics*, 18(1), 2339–2404.
> <https://doi.org/10.1214/24-EJS2254>

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API](#api)
- [File Structure](#file-structure)
- [Testing](#testing)
- [Development](#development)
- [References](#references)
- [License](#license)

## Overview

Given a high-dimensional dataset with continuous and/or ordinal variables, **humpy**
estimates an undirected graph by:

1. **Estimating a latent correlation matrix** using pair-type-specific, rank-based estimators:
   - *continuous – continuous*: Spearman sine-transform
     $\hat{\sigma} = 2\sin\bigl(\tfrac{\pi}{6}\hat{\rho}_S\bigr)$
   - *continuous – ordinal*: ad-hoc polyserial correlation
     (`PolyserialCorrelation`)
   - *ordinal – ordinal*: maximum-likelihood polychoric correlation
     (`PolychoricCorrelation`)

2. **Fitting the graphical lasso** over a log-spaced regularisation path.

3. **Selecting the sparsity level** that minimizes the extended BIC (eBIC).

4. **Returning the estimated precision matrix and a `UGRAPH`** of the
   conditional independence structure.

The design mirrors `sklearn.covariance.GraphicalLassoCV` and integrates naturally
into a scikit-learn-style workflow (`fit` returns `self`; fitted attributes carry a
trailing underscore).

## Installation

### From source

```bash
pip install humpy
```

### Development installation

```bash
git clone https://github.com/konstantingoe/mixed-gm.git
cd mixed-gm
pip install -e ".[dev]"
```

Or using Make:

```bash
make sync-venv
```

## Quick Start

Generate latent continuous data from a sparse nonparanormal model, binarize the first
half of the columns via a probit transform, then fit the mixed graphical model and
evaluate against the known truth.

```python
import numpy as np
import pandas as pd
from scipy import stats
from humpy import MixedGraphicalLasso

rng = np.random.default_rng(0)
n, d = 400, 20

# --- Sparse precision matrix ------------------------------------------------
# Identity base, signal on 12 random off-diagonal pairs, diagonal dominance
# ensures positive definiteness.
precision = np.eye(d)
pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
true_edges = set()
for idx in rng.choice(len(pairs), size=12, replace=False):
    i, j = pairs[idx]
    precision[i, j] = precision[j, i] = 0.5
    true_edges.add(frozenset((f"x{i}", f"x{j}")))
np.fill_diagonal(precision, np.abs(precision).sum(axis=1) + 0.1)

# --- Latent multivariate normal data ----------------------------------------
cov = np.linalg.inv(precision)
X = rng.multivariate_normal(np.zeros(d), cov, size=n)
X = np.sign(X) * np.power(np.abs(X), 1.5)  # nonparanormal transform

# --- Binarise first half of columns (probit / quantile transform) -----------
# Mirrors the R idiom: data[,i] <- qbinom(pnorm(scale(X[,i])), size=1, prob=p)
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

# --- Evaluation -------------------------------------------------------------
recovered = {frozenset(e) for e in mgl.graph_.edges}
tp = len(true_edges & recovered)
tpr = tp / len(true_edges)
fpr = (len(recovered) - tp) / (d * (d - 1) // 2 - len(true_edges))
print(f"TPR: {tpr:.2f}  FPR: {fpr:.2f}")
```

### Visualising the estimated graph

```python
mgl.graph_.show()
```

## API

### Core classes

#### `MixedGraphicalLasso`

Full estimation pipeline: correlation matrix → glasso path → eBIC selection →
precision matrix + `UGRAPH`.

```python
MixedGraphicalLasso(
    n_lambdas=50,        # length of the regularisation path
    ebic_gamma=0.1,      # eBIC penalty: 0 = standard BIC, higher = sparser
    n_levels_threshold=20,  # variables with fewer unique values treated as ordinal
)
```

| Fitted attribute      | Description                                                                         |
| --------------------- | ----------------------------------------------------------------------------------- |
| `precision_matrix_`   | Estimated precision matrix (partial correlations on off-diagonal) as `pd.DataFrame` |
| `correlation_matrix_` | Latent correlation matrix as `pd.DataFrame`                                         |
| `graph_`              | Estimated conditional independence graph as `UGRAPH`                                |
| `alpha_`              | Selected regularisation parameter                                                   |
| `ebic_scores_`        | Full eBIC array along the path                                                      |
| `singular_`           | `True` if the correlation matrix required PD projection                             |
| `feature_names_`      | List of variable names                                                              |
| `n_edges_`            | Number of edges (raises `RuntimeError` if not fitted)                               |

#### `SampleCorrelation`

Estimates the latent correlation matrix only, without fitting the graphical model.
Useful when you want to inspect or pre-process the correlation matrix before
running your own penalized estimator.

```python
sc = SampleCorrelation(n_levels_threshold=20).fit(data)
print(sc.correlation_matrix_)
```

### Model selection

#### `omega_select(precision_path, lambda_path, n, s, *, gamma=0.1)`

Select the best precision matrix from a pre-computed glasso path using eBIC.
Returns `(selected_omega, selected_alpha, ebic_scores)`.

### Graph classes

| Class    | Description                                                 |
| -------- | ----------------------------------------------------------- |
| `UGRAPH` | Undirected graph — the output type of `MixedGraphicalLasso` |

`UGRAPH` exposes: `nodes`, `edges`, `num_nodes`,
`num_edges`, `adjacency_matrix`, `from_pandas_adjacency()`, `to_networkx()`,
`remove_edge()`, `remove_node()`, `copy()`.

### Correlation functions

| Function / Class        | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| `PolychoricCorrelation` | MLE polychoric correlation for ordinal–ordinal pairs |
| `PolyserialCorrelation` | Ad-hoc polyserial correlation for mixed pairs        |
| `spearman(x, y)`        | Spearman's ρ                                         |
| `npn_pearson(x, y)`     | Nonparanormal Pearson correlation                    |
| `f_hat(x)`              | Nonparanormal transformation                         |

### Legacy functions

The original function-based API is preserved for backward compatibility:

```python
from humpy import mixed_graph_nonpara, mixed_graph_gauss

result = mixed_graph_nonpara(data, param=0.1)
# result.precision_matrix, result.adjacency_matrix,
# result.n_edges, result.max_degree, ...
```

## File Structure

```
mixed-gm/
├── humpy/
│   ├── __init__.py          # public API and package docstring
│   ├── estimation.py        # SampleCorrelation, MixedGraphicalLasso, omega_select
│   ├── correlation.py       # PolychoricCorrelation, PolyserialCorrelation, …
│   └── graphs.py            # UGRAPH
├── tests/
│   ├── test_estimation.py   # tests for estimation module
│   ├── test_correlation.py  # tests for correlation module
│   └── test_graphs.py       # tests for graph classes
├── docs/
│   ├── index.md             # documentation home page
│   └── reference.md         # API reference (mkdocstrings)
├── pyproject.toml
├── mkdocs.yml
├── Makefile
└── VERSION
```

## Testing

```bash
pytest tests/ -v
```

With coverage report:

```bash
pytest tests/ --cov=humpy --cov-report=html
```

Or via Make:

```bash
make test
```

## Development

### Pre-commit hooks

```bash
pre-commit install        # enable hooks in the current venv
pre-commit run --all-files  # run all hooks manually
```

### Building the documentation locally

```bash
mkdocs serve   # live preview at http://127.0.0.1:8000
mkdocs build   # static site in site/
```

### Linting and type checking

```bash
ruff check humpy/     # linting
mypy humpy/           # static type checking
```

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

## Authors

- [Konstantin Göbler](mailto:konstantin.goebler@tum.de) (author & maintainer)
- [Stephan Haug](mailto:stephan.haug@tum.de) (maintainer)

## License

MIT License — see [LICENSE](LICENSE) for details.
