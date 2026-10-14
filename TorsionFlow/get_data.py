"""Get the GEOM-QM9 data onto a new machine (the data is too big for git).

    .venv\Scripts\python.exe get_data.py

1. downloads qm9.tar.gz (770 MB) from Torsional Diffusion's shared Google Drive into data/
2. unpacks it into data/QM9/ (with Windows' built-in tar)
3. makes data/QM9/qm9_names.txt (required on Windows, see step2_reproduce/README.md)

Each part is skipped if it's already done, so it's safe to re-run.
Needs: pip install gdown
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TARBALL = DATA / "qm9.tar.gz"
DRIVE_ID = "1c75BAevYvoShxyxZZw0o4-zIQlyj_-H9"   # qm9.tar.gz in the authors' folder 1BBRpaAvvS2hTrH81mAE4WvyLIKMyhwN7
NAMES = DATA / "QM9" / "qm9_names.txt"

DATA.mkdir(exist_ok=True)

if TARBALL.exists() and TARBALL.stat().st_size > 800_000_000:
    print("1. qm9.tar.gz already downloaded")
else:
    try:
        import gdown
    except ImportError:
        sys.exit("needs gdown:  .venv\\Scripts\\python.exe -m pip install gdown")
    print("1. downloading qm9.tar.gz (770 MB) ...")
    gdown.download(id=DRIVE_ID, output=str(TARBALL), quiet=False)

if (DATA / "QM9" / "standardized_pickles").is_dir():
    print("2. already unpacked")
else:
    print("2. unpacking (a few minutes; 133k small files) ...")
    subprocess.run(["tar", "-xzf", TARBALL.name], cwd=DATA, check=True)

if NAMES.exists():
    print("3. name list already made")
else:
    print("3. making the name list ...")
    subprocess.run([sys.executable, str(ROOT / "step2_reproduce" / "make_names.py"),
                    str(TARBALL), "QM9/qm9", str(NAMES)], check=True)

print("done: data is in", DATA / "QM9")
