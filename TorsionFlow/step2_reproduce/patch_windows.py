"""
patch_windows.py - make a Torsional Diffusion checkout run on native Windows with
current library versions.

Usage:
    python patch_windows.py path\\to\\torsional-diffusion

Patches. Each one must find its original text exactly once. If the text is missing
and the patched text is already there, the patch is skipped (safe to re-run). If
neither is there, upstream changed and the script stops instead of guessing.

  1. diffusion/score_model.py  torch_cluster  -> utils/torch_radius.py (plain PyTorch)
                               torch_scatter  -> torch_geometric.utils.scatter
  2. utils/featurization.py    torch_scatter import (imported but never used) -> removed
                               torch_geometric.data.DataLoader -> torch_geometric.loader
  3. utils/dataset.py          torch_geometric.data.DataLoader -> torch_geometric.loader
                               (the old location is deprecated: in PyG 2.8 it still
                               imports, but as a separate wrapper class, not the
                               supported loader. Use the supported one.)
  4. utils/torsion.py          torch.cross(a, b) -> torch.cross(a, b, dim=-1)
                               (without dim, torch picks the first axis of size 3,
                               which is the wrong axis when the batch size is 3)
  5. evaluate_confs.py         __main__ guard around the Pool section only. The worker
                               function reads module-level globals, so the data loading
                               must stay at top level for Windows' spawned workers.
  6. copies torch_radius.py into utils/
  7. utils/dataset.py          read molecule names from <root>_names.txt (made from the
                               archive by make_names.py) instead of listing the folder.
                               The split stores positions in that list; on Windows, 121
                               case-colliding QM9 files vanish and the positions shift.
  8. utils/xtb.py              /tmp -> the system temp folder. The file creates /tmp/<pid>
                               the moment it's imported (even if xtb is never used), and
                               the sampler imports it, so generate_confs.py crashed on
                               Windows.
  9. utils/dataset.py          TorsionNoiseTransform: __call__ -> forward(), plus a small
                               __call__ that calls it. PyTorch Geometric 2.8 refuses to
                               create a transform without forward(), so train.py crashed.
"""

import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

REPLACEMENTS = [
    ("diffusion/score_model.py",
     "from torch_cluster import radius, radius_graph\n",
     "from utils.torch_radius import radius, radius_graph  # [TorsionFlow windows patch]\n"),
    ("diffusion/score_model.py",
     "from torch_scatter import scatter\n",
     "from torch_geometric.utils import scatter  # [TorsionFlow windows patch]\n"),
    ("utils/featurization.py",
     "from torch_scatter import scatter\n",
     "# [TorsionFlow windows patch] unused torch_scatter import removed\n"),
    ("utils/featurization.py",
     "from torch_geometric.data import Dataset, Data, DataLoader\n",
     "from torch_geometric.data import Dataset, Data\n"
     "from torch_geometric.loader import DataLoader  # [TorsionFlow windows patch]\n"),
    ("utils/dataset.py",
     "from torch_geometric.data import Dataset, DataLoader\n",
     "from torch_geometric.data import Dataset\n"
     "from torch_geometric.loader import DataLoader  # [TorsionFlow windows patch]\n"),
    ("utils/torsion.py",
     "torch.cross(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab)",
     "torch.cross(dshifted - c_project_ab, batch_pos[:,c] - c_project_ab, dim=-1)"),
    ("utils/xtb.py",
     "import os, subprocess\n",
     "import os, subprocess, tempfile  # [TorsionFlow windows patch]\n"),
    ("utils/xtb.py",
     'my_dir = f"/tmp/{os.getpid()}"',
     'my_dir = os.path.join(tempfile.gettempdir(), str(os.getpid()))  # [TorsionFlow windows patch] no /tmp on Windows'),
    ("utils/xtb.py",
     'path = f"/tmp/{os.getpid()}.xyz"',
     'path = os.path.join(tempfile.gettempdir(), f"{os.getpid()}.xyz")  # [TorsionFlow windows patch]'),
    ("utils/dataset.py",
     "    def __call__(self, data):\n        # select conformer",
     "    def forward(self, data):  # [TorsionFlow compat patch] newer PyG transforms implement forward()\n"
     "        # select conformer"),
    ("utils/dataset.py",
     "class TorsionNoiseTransform(BaseTransform):\n    def __init__",
     "class TorsionNoiseTransform(BaseTransform):\n"
     "    def __call__(self, data):  # [TorsionFlow compat patch] same behaviour on old and new PyG\n"
     "        return self.forward(data)\n\n"
     "    def __init__"),
    ("utils/dataset.py",
     "        smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))\n",
     "        # [TorsionFlow windows patch] The split stores POSITIONS in this sorted file list.\n"
     "        # On case-insensitive file systems (Windows), molecule files whose SMILES differ only\n"
     "        # in case overwrite each other, the list gets shorter, and positions point to the wrong\n"
     "        # molecules (on QM9: 784 of the 1000 test molecules end up in training). If\n"
     "        # <root>_names.txt exists (all names, from the archive; see make_names.py), use it.\n"
     "        _names = osp.join(osp.dirname(osp.normpath(self.root)), osp.basename(osp.normpath(self.root)) + '_names.txt')\n"
     "        if osp.exists(_names):\n"
     "            with open(_names, encoding='utf-8') as _f:\n"
     "                smiles = np.array(sorted(self.root + n + '.pickle' for n in _f.read().split('\\n') if n))\n"
     "        else:\n"
     "            smiles = np.array(sorted(glob.glob(osp.join(self.root, '*.pickle'))))\n"),
]

GUARD_ANCHOR = "random.shuffle(jobs)\n"
GUARD_LINE = "if __name__ == '__main__':  # [TorsionFlow windows patch]\n"


def replace_once(repo, rel, old, new):
    path = repo / rel
    text = path.read_text(encoding="utf-8")
    n_old, n_new = text.count(old), text.count(new)
    # Check "already patched" FIRST: a patched block can contain the original line
    # (e.g. patch 7 keeps it, indented, in its else-branch), so counting the original
    # text alone would patch the file a second time.
    if n_new >= 1:
        return "already patched"
    if n_old == 1:
        path.write_text(text.replace(old, new), encoding="utf-8")
        return "patched"
    raise SystemExit(f"STOP: {rel}: expected the original text once, found it {n_old} times "
                     f"(and the patched text {n_new} times). Upstream code changed - "
                     f"send this message to me before going further.\n  text: {old!r}")


def guard_evaluate(repo):
    """Indent everything from `random.shuffle(jobs)` to the end under a __main__ guard."""
    path = repo / "evaluate_confs.py"
    text = path.read_text(encoding="utf-8")
    if GUARD_LINE in text:
        return "already patched"
    if text.count(GUARD_ANCHOR) != 1:
        raise SystemExit("STOP: evaluate_confs.py: could not find the Pool section uniquely.")
    head, tail = text.split(GUARD_ANCHOR)
    body = GUARD_ANCHOR + tail
    indented = "".join(("    " + line) if line.strip() else line
                       for line in body.splitlines(keepends=True))
    path.write_text(head + GUARD_LINE + indented, encoding="utf-8")
    return "patched"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    repo = Path(sys.argv[1]).resolve()
    if not (repo / "diffusion" / "score_model.py").is_file():
        raise SystemExit(f"STOP: {repo} does not look like a torsional-diffusion checkout")

    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown (not a git checkout)"
    print(f"repo:   {repo}\ncommit: {commit}\n")

    for rel, old, new in REPLACEMENTS:
        print(f"  {replace_once(repo, rel, old, new):16s} {rel}")
    print(f"  {guard_evaluate(repo):16s} evaluate_confs.py (__main__ guard)")

    shutil.copy2(HERE / "torch_radius.py", repo / "utils" / "torch_radius.py")
    print(f"  {'copied':16s} utils/torch_radius.py")

    # nothing may still import the compiled extensions. torch_radius.py is exempt: its
    # self-test imports torch_cluster only optionally, to compare against it.
    leftovers = [f"{p.relative_to(repo)}:{i}" for p in repo.rglob("*.py")
                 if p.name != "torch_radius.py"
                 for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                 if line.lstrip().startswith(("import torch_scatter", "from torch_scatter",
                                              "import torch_cluster", "from torch_cluster"))]
    if leftovers:
        raise SystemExit(f"STOP: still importing compiled extensions: {leftovers}")

    for rel in ["diffusion/score_model.py", "utils/featurization.py", "utils/dataset.py",
                "utils/torsion.py", "utils/xtb.py", "evaluate_confs.py", "utils/torch_radius.py"]:
        py_compile.compile(str(repo / rel), doraise=True)

    print("\nAll patches applied, no torch_scatter/torch_cluster imports left, "
          "patched files compile.")


if __name__ == "__main__":
    main()
