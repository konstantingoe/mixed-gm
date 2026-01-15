# HUME: High-dimensional Undirected Mixed graph Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The **hume** package implements the latent Gaussian and the latent Gaussian copula modeling approaches to learning mixed high-dimensional graphs in a fast and easy-to-use manner.

This is a Python port of the [hume R package](https://github.com/konstantingoe/hume).

## Overview

Given some (high-dimensional) dataset where both discrete and continuous variables are present, the package offers functionality to estimate an undirected graph. It is assumed that all discrete variables are ordinal. If nominal variables (where the levels are unordered) are present, the user is asked to form a dummy system.

Discrete variables are assumed to have some latent continuous analogs, that are monotone transformed versions of standard Gaussians. The functions `mixed_graph_gauss()` and `mixed_graph_nonpara()` estimate the latent precision matrix of these (transformed) continuous variables where:

- **`mixed_graph_gauss()`**: Assumes no transformation (latent Gaussian model)
- **`mixed_graph_nonpara()`**: Assumes monotone and differentiable transformation functions (latent Gaussian copula model)

The latter is more general and should always be used except when the user knows that the latent variables are Gaussian (this will almost never be the case).

## Authors

- [Konstantin Göbler](mailto:konstantin.goebler@tum.de)

**Maintainer:** [Stephan Haug](mailto:stephan.haug@tum.de)

## Table of Contents

* [Documentation](#documentation)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [API Reference](#api-reference)
* [Testing](#testing)
* [Development](#development)

## <a name="documentation">Documentation</a>

Full documentation is available at [GitHub Pages](https://konstantingoe.github.io/mixed-gm/).

## <a name="installation">Installation</a>

### From source

```bash
git clone https://github.com/konstantingoe/mixed-gm.git
cd mixed-gm
pip install -e .
```

### Development installation

```bash
pip install -e ".[dev]"
```

Or using make:

```bash
make sync-venv
```

## <a name="quick-start">Quick Start</a>

```python
import numpy as np
import pandas as pd
from hume import mixed_graph_nonpara

# Create synthetic mixed data
np.random.seed(42)
n = 200  # number of observations
d = 10   # number of variables

# Generate continuous variables
continuous = np.random.randn(n, 5)

# Generate discrete (ordinal) variables
discrete = np.random.binomial(3, 0.5, (n, 5))

# Combine into DataFrame
data = pd.DataFrame(
    np.hstack([continuous, discrete]),
    columns=[f"cont_{i}" for i in range(5)] + [f"disc_{i}" for i in range(5)]
)

# Estimate the graph using nonparanormal approach (recommended)
result = mixed_graph_nonpara(data, param=0.1)

print(f"Number of edges: {result.n_edges}")
print(f"Maximum degree: {result.max_degree}")
print(f"Precision matrix shape: {result.precision_matrix.shape}")

# Access the adjacency matrix for visualization
adjacency = result.adjacency_matrix
```

### Visualization with NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt

# Create graph from adjacency matrix
G = nx.from_numpy_array(result.adjacency_matrix.astype(int))

# Relabel nodes with feature names
if result.feature_names:
    mapping = {i: name for i, name in enumerate(result.feature_names)}
    G = nx.relabel_nodes(G, mapping)

# Plot
plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=True, node_color='lightblue',
        node_size=500, font_size=10, font_weight='bold')
plt.title("Estimated Mixed Graph")
plt.show()
```

## <a name="api-reference">API Reference</a>

### Main Functions

#### `mixed_graph_nonpara(data, *, verbose=True, n_lambdas=50, param=0.1, feature_names=None, n_levels_threshold=20)`

Estimate mixed graph under nonparanormal (Gaussian copula) assumption. **Recommended for most use cases.**

**Parameters:**
- `data`: DataFrame or 2D array of shape (n, d)
- `verbose`: If True, print warnings and information
- `n_lambdas`: Length of the glasso path (default: 50)
- `param`: eBIC dimensionality penalty parameter (default: 0.1)
- `feature_names`: Optional list of feature names
- `n_levels_threshold`: Variables with fewer unique values are treated as discrete (default: 20)

**Returns:** `MixedGraphResult` dataclass containing:
- `precision_matrix`: Estimated precision matrix (partial correlations)
- `adjacency_matrix`: Boolean adjacency matrix
- `correlation_matrix`: Sample correlation matrix
- `n_edges`: Number of edges in the graph
- `max_degree`: Maximum node degree
- `initial_mat_singular`: Whether correlation matrix was singular
- `feature_names`: Node names (if provided)

#### `mixed_graph_gauss(data, **kwargs)`

Same interface as `mixed_graph_nonpara()` but uses Gaussian assumption instead of nonparanormal.

### Helper Functions

- `edgenumber(precision, *, cut=0.0)`: Count edges in precision matrix
- `omega_select(...)`: Select precision matrix using eBIC
- `spearman(x, y)`: Compute Spearman's rho
- `f_hat(x)`: Nonparanormal transformation
- `npn_pearson(cont, disc)`: Nonparanormal Pearson correlation
- `adhoc_polyserial(x, y, **kwargs)`: Adhoc polyserial correlation

## <a name="testing">Testing</a>

Run tests with pytest:

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ --cov=hume --cov-report=html
```

## <a name="development">Development</a>

### Pre-commit hooks

To use the `pre-commit` hooks, enable them in the venv:

```bash
pre-commit install
```

Run hooks for all files:

```bash
pre-commit run --all-files
```

Or use make:

```bash
make pre-commit
```

## Dependencies

- numpy >= 1.21.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0
- pandas >= 1.3.0

## References

- Foygel, Rina and Drton, Mathias. (2010). Extended Bayesian Information Criteria for Gaussian Graphical Models. *Advances in Neural Information Processing Systems*, Volume 23, pp. 604–612.

- Liu, Han, Lafferty, John and Wasserman, Larry. (2009). The nonparanormal: Semi-parametric estimation of high dimensional undirected graphs. *Journal of Machine Learning Research* 10(80), 2295–2328.

## License

MIT License - see [LICENSE](LICENSE) for details.
