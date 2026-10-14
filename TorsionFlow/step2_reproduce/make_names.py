"""
make_names.py - write every molecule name in a GEOM archive to <folder>_names.txt.

Why: Torsional Diffusion's split stores POSITIONS in the sorted list of per-molecule
files. Windows file names ignore upper/lower case, but SMILES don't (c = aromatic
carbon, C = ordinary carbon), so 121 QM9 files overwrite each other when unpacked on
Windows. The folder listing gets shorter and almost every position then points to
the wrong molecule: 784 of the 1000 official test molecules would land in training.
The archive itself still has all names, so we read them from there.
patch_windows.py (patch 7) makes the dataset use this file when it exists.

usage:  python make_names.py data/qm9.tar.gz QM9/qm9 data/QM9/qm9_names.txt
        (archive, folder inside the archive, output file)
"""

import sys
import tarfile

archive, inner, out = sys.argv[1], sys.argv[2].rstrip("/") + "/", sys.argv[3]
with tarfile.open(archive) as tf:
    names = sorted(m.name[len(inner):-len(".pickle")] for m in tf
                   if m.name.startswith(inner) and m.name.endswith(".pickle"))
if not names:
    sys.exit(f"no {inner}*.pickle entries in {archive}")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(names))
lower = {}
for n in names:
    lower.setdefault(n.lower(), []).append(n)
clashes = sum(len(v) > 1 for v in lower.values())
print(f"wrote {len(names)} names to {out}  ({clashes} pairs differ only in upper/lower case)")
