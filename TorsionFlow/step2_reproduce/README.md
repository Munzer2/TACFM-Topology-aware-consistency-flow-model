# Step 2 - Reproduce Torsional Diffusion on GEOM-QM9

Goal: get the published numbers ourselves, on Windows, before changing anything.
If we can't reproduce the baseline, nothing we compare against it later can be
trusted.

## The target (paper Table 7, GEOM-QM9 test set, threshold 0.5 A)

| Method | COV-R mean | AMR-R mean | COV-P mean | AMR-P mean |
|---|---|---|---|---|
| RDKit ETKDG | 85.1 | 0.235 | 86.8 | 0.232 |
| **Torsional Diffusion** (20 steps) | **92.8** | **0.178** | **92.7** | **0.221** |

COV = coverage, % (higher is better). AMR = average minimum RMSD, A (lower is better).
R = recall (does the generated set cover the real conformers?), P = precision (is
each generated conformer close to a real one?). For a molecule with K real
conformers, 2K are generated. Test set: 1000 molecules (split 106586 / 13323 / 1000).

**Pass condition (our tolerance, not the paper's):** AMR within 0.01 A and COV within
1.5 points of the table, for the RDKit row (2a) and the Torsional Diffusion row
(2b and/or 2c).

## Read this before step 3: the QM9 floor

The paper says Torsional Diffusion's 0.178 A on QM9 is "already very close" to a
**lower bound of about 0.17 A** that any model can reach when it keeps RDKit's bond
lengths and angles fixed and only changes torsions.

So on QM9, a torsion-only model cannot get much below 0.17 A, however good its path
or training is. The limit comes from the frozen local structure.

What this means for the project:

- **Inside the torsion family** (our few-step model vs Torsional Diffusion), QM9 is a
  fair test: same floor for both. "Can we reach ~0.178 in 1-2 steps instead of 20?"
  is a clean question here.
- **Across families** (torsion vs straight Cartesian models like AvgFlow / ET-Flow),
  QM9 is unfair to the torsion side: Cartesian models do not have this floor. That
  comparison belongs on GEOM-Drugs (step 8), where flexibility matters more.

---

## 2.0 Get the code and patch it

```powershell
cd D:\work\TorsionFlow
git clone https://github.com/gcorso/torsional-diffusion
.\.venv\Scripts\python.exe step2_reproduce\patch_windows.py torsional-diffusion
```

(No git? Download the repo as a ZIP from GitHub and extract it to
`D:\work\TorsionFlow\torsional-diffusion`.)

The script prints the commit it patched; write it down. It should end with
`All patches applied ... patched files compile.`

### What the patch changes, and how it was checked

| File | Change | Why |
|---|---|---|
| `diffusion/score_model.py` | `torch_cluster.radius/radius_graph` -> `utils/torch_radius.py` | compiled extension, often no Windows wheel |
| `diffusion/score_model.py` | `torch_scatter.scatter` -> `torch_geometric.utils.scatter` | same |
| `utils/featurization.py` | unused `torch_scatter` import removed; DataLoader import moved | same; deprecated location |
| `utils/dataset.py` | DataLoader import moved to `torch_geometric.loader` | deprecated location |
| `utils/torsion.py` | `torch.cross(..., dim=-1)` | without `dim`, picks the wrong axis when batch size is 3 |
| `evaluate_confs.py` | `__main__` guard around the Pool section | Windows starts workers by re-running the file |
| `utils/dataset.py` | read molecule names from `qm9_names.txt` instead of listing the folder | on Windows, 121 case-colliding files vanish and the split's positions shift (see 2.1) |
| `utils/xtb.py` | `/tmp` -> the system temp folder | it creates `/tmp/<pid>` when imported; the sampler imports it, so `generate_confs.py` crashed on Windows |
| `utils/dataset.py` | the noise transform gets a `forward()` method | PyTorch Geometric 2.8 refuses to create a transform without one, so `train.py` crashed |

Checks done before handing this over:

- `torch_radius.py` returns **the same edge sets as the real `torch_cluster`** (and as
  a brute-force search), including empty and single-atom molecules.
- The real Torsional Diffusion model, run on 6 QM9-sized molecules: the neighbour
  graphs (882 and 258 edges) are **identical** to `torch_cluster`'s, and the scatter
  step gives **exactly** the same values as `torch_scatter`. End-to-end outputs differ
  from the original by less than the original differs from **itself** between two
  runs (2-6e-7 vs 2e-7 on outputs of size ~1e-4). That small run-to-run variation is
  in the original code too, so it is not from the patch.
- The patch is safe to re-run: the second run reports `already patched` everywhere.

## 2.1 Get the QM9 data

The data lives in `TorsionFlow\data\`, shared by step 2, step 3 and the pipeline
notebook. On this machine it's already there: `qm9.tar.gz` (770 MB, from the authors'
shared Drive) unpacked into `data\QM9\`:

```
data\QM9\qm9\                    one .pickle per molecule (all CREST conformers)
data\QM9\split.npy               train/val/test split (positions, see below)
data\QM9\test_smiles.csv         the 1000 official test molecules
data\QM9\test_mols.pkl           their real conformers
data\QM9\standardized_pickles\   134 files: conformer matching already done
```

On another machine (the data is not in git, it's too big): run
`.\.venv\Scripts\python.exe get_data.py` from `TorsionFlow\`. It downloads `qm9.tar.gz`,
unpacks it and makes the name list below, in one go.

**Then make the name list. This is required on Windows:**

```powershell
cd D:\work\TorsionFlow
.\.venv\Scripts\python.exe step2_reproduce\make_names.py data\qm9.tar.gz QM9/qm9 data\QM9\qm9_names.txt
```

Why: the split stores **positions** in the alphabetical list of per-molecule files,
not names. SMILES are case-sensitive (`c` = aromatic carbon, `C` = ordinary carbon),
Windows file names are not, and 121 pairs of different molecules have names that
differ only in case (e.g. `O=CC(=O)n1cccc1` vs `O=CC(=O)N1CCCC1`). Unpacked on
Windows, one file of each pair overwrites the other: 133,111 files instead of
133,232. The list then shifts from position 984 onward, and **784 of the 1000 test
molecules end up in training**. Patch 7 makes the dataset read `qm9_names.txt`
(built from the archive) instead of the folder; checked: the test positions then
give exactly the 1000 official test molecules, and none of them is in training.

**Point Torsional Diffusion at the shared data** with a folder link, so every command
below works unchanged and nothing is copied:

```powershell
cd D:\work\TorsionFlow
New-Item -ItemType Junction -Path torsional-diffusion\data -Target (Resolve-Path data)
```

Conformer matching is already done (`standardized_pickles`). **Only if that folder
is missing**, run it (hours; can be stopped and resumed), and only on Linux/WSL,
because it reads the per-molecule files that Windows loses:

```powershell
cd D:\work\TorsionFlow\torsional-diffusion
..\.venv\Scripts\python.exe ..\step2_reproduce\run_matching.py --root data/QM9/qm9/ --out_dir data/QM9/standardized_pickles
```

## 2a. RDKit baseline: tests data + evaluation, no model

`--no_model --no_random` returns plain RDKit conformers, which is the paper's RDKit
row. If this matches, the data and the evaluation script are right, independent of
any training.

`generate_confs.py` always reads `<model_dir>/model_parameters.yml`, even with
`--no_model`, and takes the **dataset name** from it. So `--model_dir` must be a
**QM9** model folder: the authors' QM9 model (2b) or the folder 2c's smoke run creates
(`workdir/qm9_smoke`). A Drugs model folder would make it featurise the molecules as
Drugs. In practice: do the 2c smoke run first, then come back to this.

```powershell
cd D:\work\TorsionFlow\torsional-diffusion
..\.venv\Scripts\python.exe generate_confs.py --no_model --no_random --test_csv data/QM9/test_smiles.csv --model_dir <a model folder> --out workdir/qm9_rdkit.pkl --tqdm --no_energy
..\.venv\Scripts\python.exe evaluate_confs.py --confs workdir/qm9_rdkit.pkl --test_csv data/QM9/test_smiles.csv --true_mols data/QM9/test_mols.pkl --dataset qm9 --n_workers 8
```

`evaluate_confs.py` prints a block per threshold. **Read the `threshold 0.5` block.**
Compare with the RDKit row.

## 2b. Their trained model, if the Drive has one for QM9

Check whether the Drive's `workdir` folder contains a QM9 model (for example
`qm9_default`). If it does, put it in `torsional-diffusion\workdir\` and:

```powershell
..\.venv\Scripts\python.exe generate_confs.py --test_csv data/QM9/test_smiles.csv --inference_steps 20 --model_dir workdir/qm9_default --out workdir/qm9_default/qm9_20steps.pkl --tqdm --batch_size 128 --no_energy
..\.venv\Scripts\python.exe evaluate_confs.py --confs workdir/qm9_default/qm9_20steps.pkl --test_csv data/QM9/test_smiles.csv --true_mols data/QM9/test_mols.pkl --dataset qm9 --n_workers 8
```

Compare the `threshold 0.5` block with the Torsional Diffusion row.

## 2c. Train it ourselves

**First a smoke run** (500 molecules, 2 epochs): catches errors early and **gives the
time per epoch**, which sets how many experiments later steps can afford.

The first run also featurises the dataset and caches it (the authors quote ~2 h on
one core for Drugs; QM9 is smaller). If preprocessing errors on Windows, add
`--num_workers 1`.

```powershell
..\.venv\Scripts\python.exe train.py --log_dir workdir/qm9_smoke --dataset qm9 --in_node_features 44 --data_dir data/QM9/qm9/ --split_path data/QM9/split.npy --std_pickles data/QM9/standardized_pickles --cache data/QM9/cache --limit_train_mols 500 --n_epochs 2
```

**Then the full run**, with the paper's settings (QM9 used the same hyperparameters as
Drugs: 250 epochs, Adam, lr 1e-3, 4 layers, 5 A radius; all are the defaults):

```powershell
..\.venv\Scripts\python.exe train.py --log_dir workdir/qm9_ours --dataset qm9 --in_node_features 44 --data_dir data/QM9/qm9/ --split_path data/QM9/split.npy --std_pickles data/QM9/standardized_pickles --cache data/QM9/cache
```

Then generate and evaluate exactly as in 2b, with `--model_dir workdir/qm9_ours`.

## What to send back

1. The commit hash printed by `patch_windows.py`.
2. The folder names inside the extracted QM9 archive.
3. The `threshold 0.5` block for 2a (and 2b if available).
4. The smoke run's time per epoch.
