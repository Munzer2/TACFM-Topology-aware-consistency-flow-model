"""
fake_vs_real.py - research question 1 on real GEOM-QM9 conformers.

How much of the ambiguity a twist-angle model faces at the start of generation is
"fake" (symmetric groups whose positions are the same molecule) and how much is real?
Model-free and exact: the same calculation as part J of
notebooks/training_walkthrough.ipynb (cell 19), applied to real data.

Per rotatable bond, at t = 0 the start is uniformly random and independent of the
destination, so the best possible velocity prediction is the average move to that
bond's real angles, and the leftover error is their spread:

    ambiguity = average over starts psi0 of  Var_w[ wrap(n * phi1 - psi0) ] / n^2

with n = 1 for "360 degrees everywhere" and n = the bond's symmetry number for the
symmetry-aware version. fake = (n=1 value) - (n=symmetry value).

Two views of the result:
  all twists           what the model has to learn (the training loss sees all of it)
  heavy twists only    what the benchmark score can see. A twist moves heavy atoms
                       relative to each other only if BOTH ends have a heavy neighbour
                       besides each other. CH3 or OH spins move only hydrogens, so
                       neither their fake nor their real ambiguity can affect the score.

Molecules and conformers are chosen like Torsional Diffusion's defaults: drop SMILES
with '.', drop molecules with < 4 atoms or no rotatable bond, drop conformers whose
SMILES changed (reacted), keep at most 30 conformers, weight them uniformly (training
picks a conformer at random). A Boltzmann-weighted version is reported as a check.

usage:  python fake_vs_real.py <folder of per-molecule .pickle files> [--workers 16]
"""

import argparse
import glob
import math
import os
import pickle
import sys
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
GRID = np.linspace(-math.pi, math.pi, 721)[:-1]          # every possible start angle
MAX_CONFS = 30


def wrap(x):
    return np.remainder(x + math.pi, 2 * math.pi) - math.pi


# ---------------------------------------------------------------- same rules as the notebook
def rotatable_bonds(mol):
    n = mol.GetNumAtoms()
    nbrs = [[x.GetIdx() for x in mol.GetAtomWithIdx(i).GetNeighbors()] for i in range(n)]
    out = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        side, stack = {j}, [j]
        while stack:
            x = stack.pop()
            for y in nbrs[x]:
                if (x == j and y == i) or y in side:
                    continue
                side.add(y)
                stack.append(y)
        if i in side:                                     # ring bond
            continue
        u, v = (i, j) if len(side) <= n - len(side) else (j, i)
        swing = side if v == j else set(range(n)) - side
        if len(swing) < 2:
            continue
        a = min(x for x in nbrs[u] if x != v)
        b = min(x for x in nbrs[v] if x != u)
        out.append((u, v, a, b))
    return out


def symmetry_order(mol, x, partner, ranks):
    atom = mol.GetAtomWithIdx(x)
    others = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetIdx() != partner]
    need = {Chem.HybridizationType.SP3: 3, Chem.HybridizationType.SP2: 2}.get(atom.GetHybridization())
    if need is None or len(others) != need or len({ranks[i] for i in others}) > 1:
        return 1, None
    return need, mol.GetAtomWithIdx(others[0]).GetSymbol()


def heavy_twist(mol, u, v):
    "True if twisting u-v changes heavy-atom positions relative to each other."
    hu = any(nb.GetAtomicNum() > 1 and nb.GetIdx() != v for nb in mol.GetAtomWithIdx(u).GetNeighbors())
    hv = any(nb.GetAtomicNum() > 1 and nb.GetIdx() != u for nb in mol.GetAtomWithIdx(v).GetNeighbors())
    return hu and hv


def dihedral(pos, a, u, v, b):
    b0, b1, b2 = pos[:, a] - pos[:, u], pos[:, v] - pos[:, u], pos[:, b] - pos[:, v]
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    p = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
    q = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
    return np.arctan2((np.cross(b1, p) * q).sum(-1), (p * q).sum(-1))


def ambiguity(phi1, w, n):
    "Exact leftover error at t=0 for one twist, in rad^2 of actual twist."
    moves = wrap(n * phi1[None, :] - GRID[:, None])           # [grid, K]
    mean = (moves * w).sum(1, keepdims=True)
    return float(((moves - mean) ** 2 * w).sum(1).mean() / n ** 2)


# ---------------------------------------------------------------- one molecule
def analyse(path):
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
        if not isinstance(d, dict) or "conformers" not in d:
            return ("skip", "not a molecule pickle")
        smi = d["smiles"]
        if "." in smi or Chem.MolFromSmiles(smi) is None:
            return ("skip", "bad smiles")
        ref = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)

        confs = []
        for c in d["conformers"]:                              # drop reacted conformers
            try:
                # sanitize=False exactly as Torsional Diffusion's clean_confs; sanitizing
                # first rewrites or rejects some molecules and wrongly drops ~7% of them
                if Chem.MolToSmiles(Chem.RemoveHs(c["rd_mol"], sanitize=False), isomericSmiles=False) == ref:
                    confs.append(c)
            except Exception:
                pass
            if len(confs) == MAX_CONFS:
                break
        if not confs:
            return ("skip", "no clean conformers")
        mol = confs[0]["rd_mol"]
        if mol.GetNumAtoms() < 4:
            return ("skip", "fewer than 4 atoms")
        bonds = rotatable_bonds(mol)
        if not bonds:
            return ("skip", "no rotatable bonds")

        pos = np.stack([c["rd_mol"].GetConformer().GetPositions() for c in confs])
        w_uni = np.full((1, len(confs)), 1.0 / len(confs))
        bw = np.array([c.get("boltzmannweight", 1.0) for c in confs], dtype=float)
        w_boltz = (bw / bw.sum())[None, :]
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))

        rows = []
        for u, v, a, b in bonds:
            ku, eu = symmetry_order(mol, u, v, ranks)
            kv, ev = symmetry_order(mol, v, u, ranks)
            n = math.lcm(ku, kv)
            heavy = heavy_twist(mol, u, v)
            phi1 = dihedral(pos, a, u, v, b)
            kind = None
            if n > 1:                                          # name the symmetric group(s)
                parts = []
                for k, e, x in ((ku, eu, u), (kv, ev, v)):
                    if k > 1:
                        at = mol.GetAtomWithIdx(x)
                        ring = "ring " if at.GetIsAromatic() else ""
                        parts.append(f"{ring}{at.GetSymbol()}({e}x{k})")
                kind = "+".join(parts)
            rows.append(dict(n=n, heavy=heavy, kind=kind,
                             uni=(ambiguity(phi1, w_uni, 1), ambiguity(phi1, w_uni, n)),
                             boltz=(ambiguity(phi1, w_boltz, 1), ambiguity(phi1, w_boltz, n))))
        return ("ok", dict(smiles=smi, n_confs=len(confs), n_atoms=mol.GetNumAtoms(), bonds=rows))
    except Exception as e:                                     # report, don't crash the pool
        return ("skip", f"error: {type(e).__name__}")


# ---------------------------------------------------------------- summary
def summarise(mols, weights):
    tot = Counter()
    for m in mols:
        for r in m["bonds"]:
            full, quot = r[weights]
            sym = r["n"] > 1
            tot["all_full"] += full
            tot["all_fake"] += full - quot
            tot["fake_Honly" if not r["heavy"] else "fake_heavy"] += (full - quot) if sym else 0.0
            if r["heavy"]:
                tot["heavy_full"] += full
                tot["heavy_fake"] += (full - quot) if sym else 0.0
    return tot


def report(mols, weights, title):
    t = summarise(mols, weights)
    print(f"\n--- {title} ---")
    print(f"ALL twists (what training sees):      fake share {100 * t['all_fake'] / t['all_full']:5.1f}%")
    print(f"   of the fake part: from H-only groups {100 * t['fake_Honly'] / t['all_fake']:5.1f}%,"
          f"  from heavy-atom groups {100 * t['fake_heavy'] / t['all_fake']:5.1f}%")
    print(f"HEAVY twists only (what the score sees): fake share {100 * t['heavy_fake'] / t['heavy_full']:5.1f}%")
    print(f"   share of all ambiguity that is on heavy twists: {100 * t['heavy_full'] / t['all_full']:5.1f}%")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder with one .pickle per molecule")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--limit", type=int, default=0, help="only the first N files (0 = all)")
    ap.add_argument("--out", default=None, help="save per-molecule results to this .pkl")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "*.pickle")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"no .pickle files in {args.root}")
    print(f"{len(files)} molecule files, {args.workers} workers")

    t0 = time.time()
    with Pool(args.workers) as p:
        res = list(p.imap_unordered(analyse, files, chunksize=64))
    mols = [r for s, r in res if s == "ok"]
    skips = Counter(r for s, r in res if s == "skip")
    print(f"done in {time.time() - t0:.0f}s: {len(mols)} molecules analysed; skipped: {dict(skips)}")

    bonds = [r for m in mols for r in m["bonds"]]
    nb = len(bonds)
    print(f"\nrotatable bonds: {nb} ({nb / len(mols):.2f} per molecule); "
          f"conformers per molecule: mean {np.mean([m['n_confs'] for m in mols]):.1f}")
    print(f"   twists that move heavy atoms: {100 * sum(r['heavy'] for r in bonds) / nb:.1f}%")
    print(f"   symmetric twists: {100 * sum(r['n'] > 1 for r in bonds) / nb:.1f}%  "
          f"(H-only {100 * sum(r['n'] > 1 and not r['heavy'] for r in bonds) / nb:.1f}%, "
          f"heavy-atom {100 * sum(r['n'] > 1 and r['heavy'] for r in bonds) / nb:.1f}%)")
    has_heavy_sym = [m for m in mols if any(r["n"] > 1 and r["heavy"] for r in m["bonds"])]
    print(f"   molecules with a heavy-atom symmetric group: {len(has_heavy_sym)} "
          f"({100 * len(has_heavy_sym) / len(mols):.1f}%)")
    kinds = Counter(r["kind"] for r in bonds if r["n"] > 1 and r["heavy"])
    print("   most common heavy-atom symmetric groups: "
          + ", ".join(f"{k} {c}" for k, c in kinds.most_common(8)))

    report(mols, "uni", "uniform weights over <=30 conformers (Torsional Diffusion's default training)")
    report(mols, "boltz", "Boltzmann weights (check: does the answer depend on the weighting?)")
    if has_heavy_sym:
        report(has_heavy_sym, "uni", f"only the {len(has_heavy_sym)} molecules that have a heavy-atom symmetric group")

    if args.out:
        with open(args.out, "wb") as f:
            pickle.dump(mols, f)
        print(f"\nper-molecule results saved to {args.out}")


if __name__ == "__main__":
    main()
