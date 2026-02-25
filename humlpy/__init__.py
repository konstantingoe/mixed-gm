"""HUMLPY: High-dimensional Undirected Mixed graph Learning in PYthon.

This package implements the methodology for learning sparse undirected graphical
models from arbitrary mixed data (any combination of continuous and ordinal
variables) developed in:

    Göbler, Konstantin, Drton, Mathias, Mukherjee, Sach and Miloschewski, Anne.
    "High-dimensional undirected graphical models for arbitrary mixed data."
    Electronic Journal of Statistics 18(1): 2339–2404, 2024.
    https://doi.org/10.1214/24-EJS2254

The core idea is to map each pair of variables to a latent Pearson correlation
using pair-type-specific estimators (sine-transform of Spearman for
continuous–continuous, polyserial for continuous–ordinal, polychoric MLE for
ordinal–ordinal), then recover the graph structure via the graphical lasso with
extended BIC model selection.

Main Classes:
    SampleCorrelation: Estimate the latent sample correlation matrix.
    MixedGraphicalLasso: Fit the full sparse undirected graphical model.

Example:
    Generate data from a latent sparse Gaussian model, binarise the first half
    of the columns, then fit the mixed graphical model.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy import stats
    >>> from humlpy import MixedGraphicalLasso
    >>> rng = np.random.default_rng(0)
    >>> n, d = 400, 20
    >>> # Sparse precision matrix (identity + signal on 12 random pairs)
    >>> precision = np.eye(d)
    >>> pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    >>> true_edges = set()
    >>> for idx in rng.choice(len(pairs), size=12, replace=False):
    ...     i, j = pairs[idx]
    ...     precision[i, j] = precision[j, i] = 0.5
    ...     true_edges.add(frozenset((f"x{i}", f"x{j}")))
    >>> np.fill_diagonal(precision, np.abs(precision).sum(axis=1) + 0.1)
    >>> # Latent MVN data
    >>> cov = np.linalg.inv(precision)
    >>> X = rng.multivariate_normal(np.zeros(d), cov, size=n)
    >>> X = np.sign(X) * np.power(np.abs(X), 1.5)  # nonparanormal transform

    >>> # Binarise first half: qbinom(pnorm(scale(x)), size=1, prob=p)
    >>> n_bin = d // 2
    >>> p_bin = rng.uniform(0.4, 0.6, size=n_bin)
    >>> data = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
    >>> for i in range(n_bin):
    ...     u = stats.norm.cdf(stats.zscore(X[:, i]))
    ...     data.iloc[:, i] = stats.binom.ppf(u, n=1, p=p_bin[i])
    >>> mgl = MixedGraphicalLasso().fit(data)
    >>> print(mgl.n_edges_)
    >>> recovered = {frozenset(e) for e in mgl.graph_.edges}
    >>> tp = len(true_edges & recovered)
    >>> tpr = tp / len(true_edges)
    >>> fpr = (len(recovered) - tp) / (d * (d - 1) // 2 - len(true_edges))
    >>> print(f"TPR: {tpr:.2f}  FPR: {fpr:.2f}")

References:
    Göbler, K., Drton, M., Mukherjee, S. and Miloschewski, A. (2024).
    High-dimensional undirected graphical models for arbitrary mixed data.
    Electronic Journal of Statistics, 18(1), 2339–2404.
    https://doi.org/10.1214/24-EJS2254

    Foygel, Rina and Drton, Mathias. (2010).
    Extended Bayesian Information Criteria for Gaussian Graphical Models.
    Advances in Neural Information Processing Systems, Volume 23, pp. 604–612.

    Liu, Han, Lafferty, John and Wasserman, Larry. (2009).
    The nonparanormal: Semi-parametric estimation of high dimensional
    undirected graphs. Journal of Machine Learning Research 10(80), 2295–2328.
"""

__version__ = "0.1.0"

import logging

from humlpy.correlation import (
    adhoc_polyserial,
    f_hat,
    npn_pearson,
    spearman,
    PolychoricCorrelation,
    PolyserialCorrelation,
)

from humlpy.estimation import (
    MixedGraphResult,
    MixedGraphicalLasso,
    SampleCorrelation,
    edgenumber,
    mixed_graph_gauss,
    mixed_graph_nonpara,
    omega_select,
)

# Keep the Graph classes available
from humlpy.graphs import UGRAPH

__all__ = [
    # New class-based API
    "SampleCorrelation",
    "MixedGraphicalLasso",
    # Legacy estimation functions
    "mixed_graph_gauss",
    "mixed_graph_nonpara",
    # Legacy result container
    "MixedGraphResult",
    # Helper functions
    "edgenumber",
    "omega_select",
    # Correlation functions
    "f_hat",
    "npn_pearson",
    "spearman",
    "adhoc_polyserial",
    "PolychoricCorrelation",
    "PolyserialCorrelation",
    # Graph classes
    "UGRAPH",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
