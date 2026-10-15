"""Step 2b: the authors' trained QM9 model, scored through OUR setup (same RDKit, same scoring).

    cd D:\work\TACFM\TorsionFlow
    .\.venv\Scripts\python.exe step2_reproduce\run_td_checkpoint.py

1. downloads their model (workdir/qm9_default: best_model.pt 6.4 MB + model_parameters.yml) if missing
2. generates with their own script (generate_confs.py, their sampler) in each step count, plus plain RDKit
3. scores every set with evaluate_confs.py and prints one table

Because it runs on the same machine and RDKit as our flow model, the comparison with
notebooks/pipeline_qm9.ipynb is apples to apples (including the molecules RDKit fails on).
Results: runs/td_qm9_default/. Options: --steps 1 2 5 20  --mols 1000  --workers 8
Needs: pip install gdown (only for the download)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TD = ROOT / "torsional-diffusion"
DATA = ROOT / "data" / "QM9"
MODEL = TD / "workdir" / "qm9_default"
OUT = ROOT / "runs" / "td_qm9_default"
FILES = {"best_model.pt": "1x07OMdS5bvxOaj0q5D0bxbCICDhtG1hn",          # authors' Drive folder
         "model_parameters.yml": "1zx3ORi2zB4k7PgCJ6i1y1tTrC-3iwO_a"}   # 1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def run(cmd):
    return subprocess.run([sys.executable] + cmd, cwd=TD, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=ENV)


def score(pkl, a):
    res = run(["evaluate_confs.py", "--confs", str(pkl), "--test_csv", str(DATA / "test_smiles.csv"),
               "--true_mols", str(DATA / "test_mols.pkl"), "--dataset", "qm9",
               "--n_workers", str(a.workers), "--limit_mols", str(a.mols)])
    if "threshold 0.5\n" not in res.stdout:
        sys.exit(f"scoring failed for {pkl.name}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    block = res.stdout.split("threshold 0.5\n")[1]
    num = lambda label: float(re.search(label + r": Mean = ([\d.]+)", block).group(1))
    fails = re.search(r"(\d+) model failures", res.stdout)
    return dict(COV_R=num("Recall Coverage"), AMR_R=num("Recall AMR"), COV_P=num("Precision Coverage"),
                AMR_P=num("Precision AMR"), failed_mols=int(fails.group(1)) if fails else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, nargs="+", default=[1, 2, 5, 20])
    p.add_argument("--mols", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    MODEL.mkdir(parents=True, exist_ok=True)
    for name, gid in FILES.items():
        if not (MODEL / name).exists():
            import gdown
            print("downloading", name)
            gdown.download(id=gid, output=str(MODEL / name), quiet=True)

    common = ["--test_csv", str(DATA / "test_smiles.csv"), "--model_dir", str(MODEL),
              "--limit_mols", str(a.mols), "--batch_size", "128", "--no_energy"]
    jobs = {"RDKit (no model)": ["--no_model", "--no_random"]} | \
           {f"Torsional Diffusion, {s} step" + "s" * (s > 1): ["--inference_steps", str(s)] for s in a.steps}

    results = {}
    for label, extra in jobs.items():
        pkl = OUT / (re.sub(r"\W+", "_", label).strip("_").lower() + f"_{a.mols}mols.pkl")
        if not pkl.exists():
            t0 = time.time()
            res = run(["generate_confs.py", "--out", str(pkl)] + common + extra)
            if not pkl.exists():
                sys.exit(f"generation failed for {label}:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
            print(f"generated {label:34s} {time.time() - t0:.0f} s")
        t0 = time.time()
        results[label] = score(pkl, a)
        print(f"   scored in {time.time() - t0:.0f} s")
        json.dump(results, open(OUT / "results.json", "w"), indent=1)

    print(f"\nGEOM-QM9, first {a.mols} test molecules, threshold 0.5 A, this machine's RDKit")
    print(f"{'':36s} COV_R  AMR_R  COV_P  AMR_P  failed molecules")
    for label, r in results.items():
        print(f"{label:36s} {r['COV_R']:5.2f}  {r['AMR_R']:.3f}  {r['COV_P']:5.2f}  {r['AMR_P']:.3f}  {r['failed_mols']}")
    print("\npaper (Table 7): RDKit 85.1 / 0.235 / 86.8 / 0.232;  Torsional Diffusion 92.8 / 0.178 / 92.7 / 0.221")


if __name__ == "__main__":
    main()
