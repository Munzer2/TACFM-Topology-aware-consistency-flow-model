"""
SPECTRE-benchmark evaluation (Planar, SBM), following DiGress and DeFoG exactly.

spectre_utils.py / dist_helper.py are vendored from DiGress (MIT licence); the metric code is
unchanged. This file only wires the pieces together the way DeFoG's SpectreSamplingMetrics does:

  MMDs (compute_emd=False -> Gaussian total-variation kernel, as in DiGress/DeFoG for Planar/SBM):
      degree, clustering (100 bins, sigma 0.1), orbit (ORCA, 4-node orbits, sigma 30),
      spectre (normalized-Laplacian eigenvalues, 200 bins), wavelet (12 Abspline filters, PyGSP)
  V.U.N.: fraction of generated graphs that are unique, not isomorphic to a training graph,
      and valid (planar: connected and planar; sbm: Wald test on a fitted SBM, needs graph_tool)
  Ratios: each MMD divided by the same MMD between the training set and the reference set,
      and their average (DeFoG's "Ratio" column).

The ORCA binary is built from spectre_eval/orca/orca.cpp:
    g++ -O2 -std=c++11 -static -o orca.exe orca.cpp      (Windows; 'orca' elsewhere)
"""

from .spectre_utils import (
    compute_list_eigh,
    clustering_stats,
    degree_stats,
    eval_fraction_isomorphic,
    eval_fraction_unique_non_isomorphic_valid,
    is_planar_graph,
    is_sbm_graph,
    orbit_stats_all,
    spectral_filter_stats,
    spectral_stats,
)

MMD_KEYS = ("degree", "clustering", "orbit", "spectre", "wavelet")


def mmd_metrics(ref_graphs, gen_graphs, keys=MMD_KEYS):
    out = {}
    if "degree" in keys:
        out["degree"] = degree_stats(ref_graphs, gen_graphs, is_parallel=True, compute_emd=False)
    if "clustering" in keys:
        out["clustering"] = clustering_stats(ref_graphs, gen_graphs, bins=100, is_parallel=True, compute_emd=False)
    if "orbit" in keys:
        out["orbit"] = orbit_stats_all(ref_graphs, gen_graphs, compute_emd=False)
    if "spectre" in keys:
        out["spectre"] = spectral_stats(ref_graphs, gen_graphs, is_parallel=True, n_eigvals=-1, compute_emd=False)
    if "wavelet" in keys:
        ref_vals, ref_vecs = compute_list_eigh(ref_graphs)
        gen_vals, gen_vecs = compute_list_eigh(gen_graphs)
        out["wavelet"] = spectral_filter_stats(ref_vecs, ref_vals, gen_vecs, gen_vals,
                                               is_parallel=False, compute_emd=False)
    return out


def validity_fn(dataset):
    if dataset == "planar":
        return is_planar_graph
    if dataset == "sbm":
        return is_sbm_graph
    raise ValueError(dataset)


def vun_metrics(gen_graphs, train_graphs, dataset, check_validity=True):
    valid_fn = validity_fn(dataset) if check_validity else (lambda g: True)
    uniq, uniq_novel, uniq_novel_valid = eval_fraction_unique_non_isomorphic_valid(
        gen_graphs, train_graphs, valid_fn)
    out = {"unique": uniq, "unique_novel": uniq_novel,
           "novel": 1.0 - eval_fraction_isomorphic(gen_graphs, train_graphs)}
    if check_validity:
        out["valid"] = sum(bool(valid_fn(g)) for g in gen_graphs) / len(gen_graphs)
        out["vun"] = uniq_novel_valid
    return out


def ratios(gen_mmd, ref_mmd):
    """DeFoG's compute_ratios: reference rounded to 4 decimals, zero references skipped."""
    out = {}
    for k, v in gen_mmd.items():
        r = round(ref_mmd.get(k, 0.0), 4)
        if r != 0.0:
            out[k + "_ratio"] = v / r
    out["average_ratio"] = sum(out.values()) / len(out) if out else -1
    return out
