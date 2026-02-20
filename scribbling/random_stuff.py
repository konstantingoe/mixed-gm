"""Quick testing."""

import numpy as np
from hume.correlation import PolychoricCorrelation, PolyserialCorrelation, adhoc_polyserial

# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    z = rng.multivariate_normal(np.zeros(2), [[1, 0.4], [0.4, 1]], size=50000)
    x_bin = (z[:, 0] > 0).astype(int)
    y_bin = (z[:, 1] > 0).astype(int)
    print("Polychoric:", PolychoricCorrelation().fit(x_bin, y_bin).correlation)

    x_cont = z[:, 0]
    y_ord = (z[:, 1] > -1).astype(int) + (z[:, 1] > 0).astype(int) + (z[:, 1] > 1).astype(int)
    print("Polyserial:", PolyserialCorrelation().fit(x_cont, y_ord).correlation)
    print("Dispatcher:", adhoc_polyserial(x_cont, y_ord))
