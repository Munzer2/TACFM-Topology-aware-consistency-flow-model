"""
run_matching.py - run Torsional Diffusion's conformer matching (standardize_confs.py)
on Windows, several workers at a time.

Only needed if the downloaded QM9 archive does NOT already contain the
`standardized_pickles` folder.

The upstream README launches workers with a bash loop
(`for i in $(seq 0 299); do python standardize_confs.py ... & done`), which does not
run in PowerShell. This does the same thing in Python.

Run from inside the torsional-diffusion folder:
    python <path>\\run_matching.py --root data/QM9/qm9/ --out_dir data/QM9/standardized_pickles

Resumable: a worker whose output file already exists is skipped, so after a crash or
a reboot just run the same command again.
"""

import argparse
import glob
import math
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="folder of per-molecule .pickle files, ending in /")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--jobs_per_worker", type=int, default=1000)
    p.add_argument("--confs_per_mol", type=int, default=30)
    p.add_argument("--parallel", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="workers running at once (default: half the CPU cores)")
    args = p.parse_args()

    if not os.path.isfile("standardize_confs.py"):
        sys.exit("run this from inside the torsional-diffusion folder")
    if not args.root.endswith(("/", "\\")):
        sys.exit("--root must end with a slash: standardize_confs.py globs f'{root}*.pickle'")

    n_files = len(glob.glob(f"{args.root}*.pickle"))
    if n_files == 0:
        sys.exit(f"no *.pickle files found under {args.root}")
    n_workers = math.ceil(n_files / args.jobs_per_worker)
    os.makedirs(args.out_dir, exist_ok=True)

    # standardize_confs.py names its output str(worker_id).zfill(3) + '.pickle'
    todo = [i for i in range(n_workers)
            if not os.path.exists(os.path.join(args.out_dir, f"{i:03d}.pickle"))]
    print(f"{n_files} molecules -> {n_workers} workers of {args.jobs_per_worker}; "
          f"{n_workers - len(todo)} already done, {len(todo)} to run, {args.parallel} at a time\n")

    running, failed, start = [], [], time.time()
    while todo or running:
        while todo and len(running) < args.parallel:
            i = todo.pop(0)
            log = open(os.path.join(args.out_dir, f"{i:03d}.log"), "w")
            cmd = [sys.executable, "standardize_confs.py",
                   "--root", args.root, "--out_dir", args.out_dir,
                   "--worker_id", str(i), "--jobs_per_worker", str(args.jobs_per_worker),
                   "--confs_per_mol", str(args.confs_per_mol)]
            running.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(2)
        for item in running[:]:
            i, proc, log = item
            if proc.poll() is None:
                continue
            log.close()
            running.remove(item)
            if proc.returncode != 0:
                failed.append(i)
            mins = (time.time() - start) / 60
            status = "ok" if proc.returncode == 0 else f"FAILED, see {i:03d}.log"
            print(f"  worker {i:03d} {status}   ({mins:.0f} min elapsed, "
                  f"{len(todo)} queued, {len(running)} running)")

    if failed:
        sys.exit(f"\n{len(failed)} workers failed: {failed}. Re-running retries only those.")
    print("\nconformer matching complete")


if __name__ == "__main__":
    main()
