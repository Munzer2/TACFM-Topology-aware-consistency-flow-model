"""Score the current best_model.pt while training is still running.

Run it from a second PowerShell window (training in Jupyter keeps going):

    cd D:\work\TACFM\TorsionFlow
    .\.venv\Scripts\python.exe score_checkpoint.py

It copies best_model.pt (so training can keep overwriting it), generates conformers for the
first --mols test molecules exactly like the notebook (sections 5-6), and scores them with the
benchmark's own evaluate_confs.py. Two baselines are scored on the SAME molecules:
  "RDKit (no model)"   plain RDKit conformers, i.e. the paper's RDKit row
  "random twists"      RDKit shapes with every twist randomised (our model's starting point)
Results go to runs/<run>/checks/<time>/; the notebook's own files are not touched.

Options: --mols 100  --steps 1 2 5 20  --mode full|smoke  --workers 4
"""
import argparse
import copy
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent
TD = ROOT / "torsional-diffusion"
DATA = ROOT / "data" / "QM9"
SIZES = {"smoke": dict(ns=16, nv=4, layers=2), "full": dict(ns=32, nv=8, layers=4)}   # as in the notebook
SIGMA_MIN, SIGMA_MAX = 0.01 * np.pi, np.pi
SEED, EMBED_SEED, BATCH = 0, 42, 32                                                    # as in the notebook


def time_as_sigma(t):
    return SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="full", choices=SIZES)
    p.add_argument("--run", default=None, help="run folder (default runs/qm9_flow_<mode>)")
    p.add_argument("--mols", type=int, default=100, help="first N official test molecules")
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 20])
    p.add_argument("--workers", type=int, default=4, help="CPU workers for scoring (training also needs CPU)")
    a = p.parse_args()
    run = Path(a.run) if a.run else ROOT / "runs" / f"qm9_flow_{a.mode}"
    out_dir = run / "checks" / datetime.now().strftime("%Y-%m-%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog("rdApp.*")

    # --- import Torsional Diffusion from inside its folder (it keeps lookup tables there) ---
    sys.path.insert(0, str(TD))
    here = os.getcwd(); os.chdir(TD)
    from utils.utils import get_model
    from diffusion.sampling import get_seed, embed_seeds, perturb_seeds, pyg_to_mol, InferenceDataset
    os.chdir(here)

    # --- snapshot the checkpoint; training may be writing it right now, so retry ---
    ckpt = out_dir / "model_snapshot.pt"
    for attempt in range(10):
        try:
            shutil.copyfile(run / "best_model.pt", ckpt)
            state = torch.load(ckpt, map_location="cpu")
            break
        except Exception as e:
            if attempt == 9:
                raise
            print(f"checkpoint busy ({type(e).__name__}), retrying in 5 s")
            time.sleep(5)
    saved_at = datetime.fromtimestamp((run / "best_model.pt").stat().st_mtime)
    print(f"model: {run / 'best_model.pt'}  (saved {saved_at:%Y-%m-%d %H:%M})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    s = SIZES[a.mode]
    model = get_model(Namespace(
        in_node_features=44, in_edge_features=4, ns=s["ns"], nv=s["nv"], sigma_embed_dim=32,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, num_conv_layers=s["layers"], max_radius=5.0,
        radius_embed_dim=50, scale_by_sigma=False, use_second_order_repr=False,
        no_residual=False, no_batch_norm=False)).to(device)
    model.load_state_dict(state)
    model.eval()
    test = pd.read_csv(DATA / "test_smiles.csv").head(a.mols)
    print(f"device {device}, {len(test)} test molecules, results -> {out_dir}\n")

    def embed(mol, n):
        AllChem.EmbedMultipleConfs(mol, numConfs=n, randomSeed=EMBED_SEED)
        return mol

    @torch.no_grad()
    def generate(smi, n_confs, steps):
        "steps = None: plain RDKit. 0: random twists. S >= 1: our model in S Euler steps."
        mol, data = get_seed(smi, dataset="qm9")
        if not mol:
            return None
        confs, _ = embed_seeds(mol, data, n_confs, embed_func=embed)
        if not confs:
            return None
        if steps is not None and int(data.edge_mask.sum()) > 0:
            confs = perturb_seeds(confs)
            ds = InferenceDataset(confs)
            for i in range(steps):
                for batch in DataLoader(ds, batch_size=BATCH, shuffle=False):
                    b = copy.deepcopy(batch).to(device)
                    b.node_sigma = time_as_sigma(i / steps) * torch.ones(b.num_nodes, device=device)
                    velocity = model(b).edge_pred.cpu().numpy()
                    ds.apply_torsion_and_update_pos(batch, velocity / steps)
        return [pyg_to_mol(mol, c, rmsd=False) for c in confs]

    def generate_all(steps, name):
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)   # same starts for every S
        out, t0 = {}, time.time()
        for _, k, smi in test.itertuples(index=False):
            mols = generate(smi, 2 * k, steps)
            if mols:
                out[smi] = mols
        path = out_dir / f"{name}.pkl"
        pickle.dump(out, open(path, "wb"))
        print(f"generated {name:12s} {len(out)}/{len(test)} molecules in {time.time() - t0:.0f} s")
        return path

    def evaluate(path):
        cmd = [sys.executable, "evaluate_confs.py", "--confs", str(path),
               "--test_csv", str(DATA / "test_smiles.csv"), "--true_mols", str(DATA / "test_mols.pkl"),
               "--dataset", "qm9", "--n_workers", str(a.workers), "--limit_mols", str(a.mols)]
        res = subprocess.run(cmd, cwd=TD, capture_output=True, text=True, encoding="utf-8", errors="replace",
                             env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        if "threshold 0.5\n" not in res.stdout:
            sys.exit(f"scoring failed for {path.name}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
        block = res.stdout.split("threshold 0.5\n")[1]
        num = lambda label: float(re.search(label + r": Mean = ([\d.]+)", block).group(1))
        return dict(COV_R=num("Recall Coverage"), AMR_R=num("Recall AMR"),
                    COV_P=num("Precision Coverage"), AMR_P=num("Precision AMR"))

    rows = {"RDKit (no model)": None, "random twists": 0} | {f"ours, {S} step" + "s" * (S > 1): S for S in a.steps}
    results = {}
    for label, steps in rows.items():
        name = "rdkit" if steps is None else f"{steps}steps"
        path = generate_all(steps, name)
        t0 = time.time()
        results[label] = evaluate(path)
        print(f"   scored in {time.time() - t0:.0f} s")

    table = pd.DataFrame(results).T
    json.dump(dict(model_saved=str(saved_at), mols=a.mols, results=results),
              open(out_dir / "results.json", "w"), indent=1)
    print(f"\nGEOM-QM9, first {a.mols} test molecules, threshold 0.5 A "
          f"(COV in %, higher is better; AMR in A, lower is better)")
    print(table.round(3).to_string())
    print("\nFor reference, the paper on all 1000 molecules: RDKit AMR-R 0.235, "
          "Torsional Diffusion (20 steps) AMR-R 0.178.\nCompare our rows with the RDKit row ABOVE, "
          "which is on the same molecules.")


if __name__ == "__main__":
    main()
