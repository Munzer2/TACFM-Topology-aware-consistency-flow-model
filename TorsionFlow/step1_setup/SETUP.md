# Step 1 - Environment on the RTX 5080 (native Windows, no WSL)

Goal: PyTorch sees the 5080, and every library Torsional Diffusion needs imports and
runs on the GPU.

All commands are **PowerShell**.

---

## Why native Windows is fine here

Torsional Diffusion has only three things that are awkward on Windows:

| Problem | Where | Fix |
|---|---|---|
| `torch_scatter` (compiled extension) | `diffusion/score_model.py`, `utils/featurization.py` | use `torch_geometric.utils.scatter`, which runs on plain PyTorch |
| `torch_cluster` (compiled extension) | `diffusion/score_model.py` (`radius`, `radius_graph`) | a pure-PyTorch neighbour search (we write it in step 2) |
| multiprocessing without a `__main__` guard | `evaluate_confs.py` | add the guard (3 lines, step 2) |

So we **do not install** `torch_scatter` or `torch_cluster` at all. Those two are the
packages that most often have no prebuilt Windows wheel for a new GPU, so skipping
them removes the riskiest part of the install.

Later, AvgFlow (step 7) is written in JAX, which has no GPU support on native
Windows. We only need AvgFlow as a comparison, so we use its released checkpoints
and published numbers. Running its 1-2 step sampler on the CPU is enough for that.

---

## A. NVIDIA driver

Install or update the latest NVIDIA driver (Game Ready or Studio). The 5080 needs a
recent one.

Check it:

```powershell
nvidia-smi
```

You should see the RTX 5080.

## B. Python 3.11

If the machine does not have it, install **Python 3.11** from python.org (64-bit).
Check:

```powershell
py -3.11 --version
```

## C. Put the project on a plain local folder

Copy this `TorsionFlow` folder somewhere like `D:\work\TorsionFlow`.

Avoid OneDrive or Dropbox folders. Sync software locks files and slows down
datasets and checkpoints.

## D. Create a virtual environment

```powershell
cd D:\work\TorsionFlow
py -3.11 -m venv .venv
```

We call the venv's Python directly (`.\.venv\Scripts\python.exe`) instead of
"activating" the venv. Activation can be blocked by PowerShell's script policy, and
calling Python directly sidesteps that.

## E. PyTorch built for CUDA 12.8 or newer

The 5080 is a Blackwell card. It needs a PyTorch build made with **CUDA 12.8 or
newer**. Older builds install fine and then fail at runtime with
"no kernel image is available".

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

If pytorch.org's Get Started page lists a newer CUDA build (cu129, cu130, ...), that
also works. Anything **below** cu128 will not.

## F. Everything else

```powershell
.\.venv\Scripts\python.exe -m pip install torch_geometric e3nn rdkit spyrmsd pyyaml matplotlib pandas tqdm rmsd gdown
```

**Do not** install `torch_scatter`, `torch_cluster` or `torch_sparse`. If some guide
or error message tells you to, ignore it and tell me instead.

For the notebook (`notebooks/training_walkthrough.ipynb`), install Jupyter too:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyterlab ipykernel
.\.venv\Scripts\python.exe -m jupyter lab
```

That opens Jupyter in your browser; open the notebook from the file list on the
left and run cells one at a time with Shift+Enter. (VS Code works too: open the
`.ipynb` file and pick `.venv` as the kernel.)

## G. Run the check

```powershell
cd D:\work\TorsionFlow\step1_setup
..\.venv\Scripts\python.exe check_env.py
```

**Step 1 passes when every line says PASS.**

Send me the full output either way. If something fails, the output says which
library and why, and we fix it before step 2.
