"""HUME: High-dimensional Undirected Mixed graph Estimation.

This package implements the latent Gaussian and latent Gaussian copula
modeling approaches to learning mixed high-dimensional graphs in a fast
and easy-to-use manner.

The package can be applied on continuous only (then it boils down to the
nonparanormal SKEPTIC) and any mix of discrete and continuous variables.

Main Functions:
    mixed_graph_gauss: Estimate graph under Gaussian assumption.
    mixed_graph_nonpara: Estimate graph under nonparanormal assumption (recommended).

Example:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from hume import mixed_graph_nonpara
    >>> # Create mixed data
    >>> n, d = 100, 5
    >>> continuous = np.random.randn(n, 3)
    >>> discrete = np.random.binomial(1, 0.5, (n, 2))
    >>> data = pd.DataFrame(np.hstack([continuous, discrete]))
    >>> result = mixed_graph_nonpara(data, verbose=False)

References:
    Foygel, Rina and Drton, Mathias. (2010).
    Extended Bayesian Information Criteria for Gaussian Graphical Models.
    Advances in Neural Information Processing Systems, Volume 23, pp. 604–612.

    Liu, Han, Lafferty, John and Wasserman, Larry. (2009).
    The nonparanormal: Semi-parametric estimation of high dimensional
    undirected graphs. Journal of Machine Learning Research 10(80), 2295–2328.
"""

import logging

from hume.correlation import (
    adhoc_polyserial,
    f_hat,
    npn_pearson,
    spearman,
)
from hume.estimation import (
    MixedGraphResult,
    edgenumber,
    mixed_graph_gauss,
    mixed_graph_nonpara,
    omega_select,
)

# Keep the Graph class for backwards compatibility
from hume.graphs import Graph

__all__ = [
    # Main estimation functions
    "mixed_graph_gauss",
    "mixed_graph_nonpara",
    # Result container
    "MixedGraphResult",
    # Helper functions
    "edgenumber",
    "omega_select",
    # Correlation functions
    "spearman",
    "f_hat",
    "npn_pearson",
    "adhoc_polyserial",
    # Graph class (legacy)
    "Graph",
]

logging.getLogger(__name__).addHandler(logging.NullHandler())
