"""Mini benchmark: polychoric solver comparison (Newton vs Brent).

Run with:
    python scribbling/random_stuff.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from hume import MixedGraphicalLasso

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
if mgl.graph_ is not None:
    print(f"Selected alpha:  {mgl.alpha_:.4f}")
    print(f"Number of edges: {mgl.n_edges_}")
    print(f"Edges:           {mgl.graph_.edges}")

    # --- Evaluation -------------------------------------------------------------
    recovered = {frozenset(e) for e in mgl.graph_.edges}
    tp = len(true_edges & recovered)
    tpr = tp / len(true_edges)
    fpr = (len(recovered) - tp) / (d * (d - 1) // 2 - len(true_edges))
    print(f"TPR: {tpr:.2f}  FPR: {fpr:.2f}")
