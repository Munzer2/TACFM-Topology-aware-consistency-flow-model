"""
prepare_spectre.py — SPECTRE Planar / SBM data with the DiGress / DeFoG splits.

Raw files (Martinkus et al., SPECTRE, MIT licence), downloaded to data/spectre/raw/:
    https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt
    https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt

Split: identical to DiGress src/datasets/spectre_dataset.py (DeFoG uses the same code):
    torch.Generator().manual_seed(0); indices = torch.randperm(200, generator=g)
    test 40 = round(200 * 0.2), train 128 = round(160 * 0.8), val 32
    train = indices[:128], val = indices[128:160], test = indices[160:]

Writes data/spectre/{name}_{train,val,test}.pkl (lists of networkx graphs, node order as
stored in the raw file) and data/spectre/{name}_split.json (the raw indices of each split).

Run from Codes/SBM:  python -I data/prepare_spectre.py
"""

import json
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RAW = HERE / "spectre" / "raw"
OUT = HERE / "spectre"
FILES = {"planar": "planar_64_200.pt", "sbm": "sbm_200.pt"}


def digress_split(num_graphs=200):
    g = torch.Generator()
    g.manual_seed(0)
    test_len = int(round(num_graphs * 0.2))
    train_len = int(round((num_graphs - test_len) * 0.8))
    val_len = num_graphs - train_len - test_len
    idx = torch.randperm(num_graphs, generator=g)
    return {"train": idx[:train_len].tolist(),
            "val": idx[train_len:train_len + val_len].tolist(),
            "test": idx[train_len + val_len:].tolist()}


def main():
    for name, fname in FILES.items():
        # weights_only=True: the file is only tensors/lists/numbers, so no arbitrary unpickling.
        raw = torch.load(RAW / fname, weights_only=True)
        adjs = raw[0]
        graphs = []
        for a in adjs:
            a = a.numpy() if torch.is_tensor(a) else np.asarray(a)
            G = nx.from_numpy_array((a > 0).astype(np.int8))
            G.remove_edges_from(nx.selfloop_edges(G))
            graphs.append(G)

        split = digress_split(len(graphs))
        for part, ids in split.items():
            pickle.dump([graphs[i] for i in ids], open(OUT / f"{name}_{part}.pkl", "wb"))
        json.dump(split, open(OUT / f"{name}_split.json", "w"))

        n = [G.number_of_nodes() for G in graphs]
        m = [G.number_of_edges() for G in graphs]
        print(f"{name}: {len(graphs)} graphs | nodes {min(n)}-{max(n)} (mean {np.mean(n):.1f}) | "
              f"edges {min(m)}-{max(m)} (mean {np.mean(m):.1f}) | "
              f"split {len(split['train'])}/{len(split['val'])}/{len(split['test'])} | "
              f"connected {sum(nx.is_connected(G) for G in graphs)}/{len(graphs)}")
        if name == "planar":
            ok = sum(nx.check_planarity(G)[0] for G in graphs)
            print(f"  planar: {ok}/{len(graphs)}")


if __name__ == "__main__":
    main()
