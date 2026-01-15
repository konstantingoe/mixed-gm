# HUME: High-dimensional Undirected Mixed graph Estimation

This site contains the documentation for the `hume` Python package.

## Overview

The **hume** package implements the latent Gaussian and the latent Gaussian copula
modeling approaches to learning mixed high-dimensional graphs.

Given some (high-dimensional) dataset where both discrete and continuous variables
are present, the package offers functionality to estimate an undirected graph.

## Quick Start

```python
import pandas as pd
from hume import mixed_graph_nonpara

# Load your mixed data
data = pd.read_csv("your_data.csv")

# Estimate the graph
result = mixed_graph_nonpara(data)

# Access results
print(f"Number of edges: {result.n_edges}")
print(f"Adjacency matrix shape: {result.adjacency_matrix.shape}")
```

## Installation

```bash
pip install -e .
```

## Main Functions

- `mixed_graph_nonpara()`: Estimate graph under nonparanormal assumption (recommended)
- `mixed_graph_gauss()`: Estimate graph under Gaussian assumption

## References

See the [API Reference](reference.md) for detailed documentation.
