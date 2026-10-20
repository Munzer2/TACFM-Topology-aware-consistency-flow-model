"""
metrics_v2.py — MMD metrics with the kernel-saturation bug in evaluate.py fixed.

THE BUG
-------
evaluate.py's gaussian_emd() computes EMD on the raw bin-index scale and drops the
`distance_scaling` argument that the reference GDSS/GraphRNN implementation passes.
For clustering (100 bins, sigma=0.1) the EMD between two *real* graphs is ~9.6, so
the kernel is exp(-9.6^2 / 0.02) = exp(-4608) = 0 in float64. Every kernel value
underflows, so kxx = kyy = kxy = 0 and MMD collapses to 0 for ANY input.

Consequence: clustering MMD in evaluate.py does not measure similarity to the
reference at all. It measures the generated set's internal self-similarity (kyy).
Comparing real graphs against random G(16,0.5) scores ~1e-44 ("perfect"); comparing
against path graphs scores 1.0 ("worst"). Both are meaningless.

THE FIX
-------
Divide the EMD by `distance_scaling` before the kernel, matching the reference
implementation which calls compute_mmd(..., sigma=1.0/10, distance_scaling=bins).

The 1D EMD itself (sum of |CDF difference|) is correct and equivalent to the
reference's pyemd call with a toeplitz distance matrix, so that part is unchanged.

Run this file directly to see the known-answer validation.
"""

import numpy as np
import networkx as nx
from scipy.linalg import eigvalsh


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """Gaussian kernel over the 1D Earth Mover's Distance between two histograms."""
    if len(x) < len(y):
        x = np.concatenate([x, np.zeros(len(y) - len(x))])
    elif len(y) < len(x):
        y = np.concatenate([y, np.zeros(len(x) - len(y))])

    x = x / (x.sum() + 1e-8)
    y = y / (y.sum() + 1e-8)

    # 1D EMD = L1 distance between CDFs, on a unit-spaced support.
    emd = np.sum(np.abs(np.cumsum(x) - np.cumsum(y)))
    emd = emd / distance_scaling  # <-- the fix
    return float(np.exp(-emd * emd / (2 * sigma * sigma)))


def compute_mmd(samples_ref, samples_pred, sigma=1.0, distance_scaling=1.0):
    """Unbiased MMD^2 estimator (diagonal excluded, so it may go slightly negative)."""
    n, m = len(samples_ref), len(samples_pred)
    if n < 2 or m < 2:
        return 1.0

    def _k(a, b):
        return gaussian_emd(a, b, sigma, distance_scaling)

    kxx = sum(_k(samples_ref[i], samples_ref[j])
              for i in range(n) for j in range(i + 1, n)) / (n * (n - 1) / 2)
    kyy = sum(_k(samples_pred[i], samples_pred[j])
              for i in range(m) for j in range(i + 1, m)) / (m * (m - 1) / 2)
    kxy = sum(_k(samples_ref[i], samples_pred[j])
              for i in range(n) for j in range(m)) / (n * m)
    return kxx + kyy - 2 * kxy


def degree_stats(ref, pred):
    a = [np.array(nx.degree_histogram(G)) for G in ref if G.number_of_nodes() > 0]
    b = [np.array(nx.degree_histogram(G)) for G in pred if G.number_of_nodes() > 0]
    if not b:
        return 1.0
    return compute_mmd(a, b, sigma=1.0, distance_scaling=1.0)


def clustering_stats(ref, pred, bins=100):
    def hist(G):
        cc = list(nx.clustering(G).values())
        h, _ = np.histogram(cc, bins=bins, range=(0.0, 1.0))
        return h

    a = [hist(G) for G in ref if G.number_of_nodes() > 0]
    b = [hist(G) for G in pred if G.number_of_nodes() > 0]
    if not b:
        return 1.0
    # Reference impl: sigma=1/10 with distance_scaling=bins.
    return compute_mmd(a, b, sigma=1.0 / 10, distance_scaling=bins)


def _raw_emd(x, y):
    if len(x) < len(y):
        x = np.concatenate([x, np.zeros(len(y) - len(x))])
    elif len(y) < len(x):
        y = np.concatenate([y, np.zeros(len(x) - len(y))])
    x = x / (x.sum() + 1e-8)
    y = y / (y.sum() + 1e-8)
    return np.sum(np.abs(np.cumsum(x) - np.cumsum(y)))


def median_heuristic_sigma(hists):
    """Standard MMD bandwidth choice: sigma = median pairwise distance.

    Needed for spectral, which is not a GDSS metric (GDSS reports Orbit) and so has
    no published sigma to inherit. With bins=200 and sigma=1 the kernel sits at
    0.9994 between real graphs -- saturated at the top, and just as uninformative as
    the underflow at the bottom. The median heuristic puts the median kernel at
    exp(-0.5) ~ 0.61 by construction. Computed from the REFERENCE set only, so the
    metric does not shift between models under comparison.
    """
    d = [_raw_emd(hists[i], hists[j])
         for i in range(len(hists)) for j in range(i + 1, len(hists))]
    med = float(np.median(d))
    return med if med > 1e-8 else 1.0


def spectral_stats(ref, pred, bins=200, sigma=None):
    def hist(G):
        if G.number_of_nodes() < 2:
            return None
        try:
            eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
        except Exception:
            return None
        h, _ = np.histogram(eigs, bins=bins, range=(-1e-5, 2))
        return h

    a = [h for h in (hist(G) for G in ref) if h is not None]
    b = [h for h in (hist(G) for G in pred) if h is not None]
    if not b or len(a) < 2:
        return 1.0
    if sigma is None:
        sigma = median_heuristic_sigma(a)
    return compute_mmd(a, b, sigma=sigma, distance_scaling=1.0)


if __name__ == "__main__":
    import pickle
    import warnings
    warnings.filterwarnings("ignore")

    test = pickle.load(open("data/community_v2_test.pkl", "rb"))[:30]
    train = pickle.load(open("data/community_v2_train.pkl", "rb"))[:30]

    # Known-answer cases. A working metric must ORDER these correctly:
    #   same-distribution  <  wrong-density  <  totally-wrong-structure
    cases = [
        ("identical set (expect ~0)", test),
        ("same distribution, disjoint (expect ~0)", train),
        ("G(16,0.5) random (expect >0)", [nx.gnp_random_graph(16, 0.5, seed=s) for s in range(30)]),
        ("path graphs (expect large)", [nx.path_graph(16) for _ in range(30)]),
        ("complete graphs (expect large)", [nx.complete_graph(16) for _ in range(30)]),
    ]

    import evaluate as old

    print(f"{'case':<42}{'metric':<8}{'OLD (buggy)':>14}{'NEW (fixed)':>14}")
    print("-" * 78)
    for name, pred in cases:
        for label, fn_new, fn_old in [
            ("deg", degree_stats, old.degree_stats),
            ("clus", clustering_stats, old.clustering_stats),
            ("spec", spectral_stats, old.spectral_stats),
        ]:
            print(f"{name if label=='deg' else '':<42}{label:<8}"
                  f"{fn_old(test, pred):>14.6f}{fn_new(test, pred):>14.6f}")
        print()
