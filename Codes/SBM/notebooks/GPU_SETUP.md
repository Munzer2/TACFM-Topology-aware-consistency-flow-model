# Running `spectre_gpu_training.ipynb` in VS Code on the RTX 5080

These steps set up a Python virtual environment (venv) and run the training notebook inside VS Code. Run every terminal command in **VS Code's terminal** (Terminal → New Terminal), from the **repository root** (the folder that contains `Codes/`).

## 1. Before you start

You need:
- **NVIDIA driver 570 or newer.** The RTX 5080 is a Blackwell GPU and needs a CUDA 12.8–capable driver. Check with `nvidia-smi`: the top right should show `CUDA Version: 12.8` or higher.
- **Python 3.11 or 3.12.** Check with `python --version` (Windows) or `python3 --version` (Linux).
- **git**
- **VS Code** with two extensions from the Extensions panel (Ctrl+Shift+X):
  - **Python** (Microsoft)
  - **Jupyter** (Microsoft)
- **Linux only:** a C++ compiler for the ORCA tool (`sudo apt install build-essential`). Windows uses the prebuilt `orca.exe` in the repo.

## 2. Get the code and open it in VS Code

If the repository is already on the machine, open its folder in VS Code (File → Open Folder…) and run:

```bash
git pull
```

Otherwise, clone it:

```bash
git clone https://github.com/Munzer2/TACFM-Topology-aware-consistency-flow-model.git
```

Then open the cloned folder in VS Code (File → Open Folder…). Open the **repository root**, not `Codes/SBM`, so the venv and the paths below line up.

## 3. Create the virtual environment

The venv lives in `.venv/` at the repository root. git ignores it. VS Code detects a `.venv` folder in the open workspace automatically.

**Windows (PowerShell, VS Code's default terminal):**

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the activate script, allow local scripts for your user once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Linux:**

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

Your terminal prompt should now start with `(.venv)`. Every step below assumes the venv is active.

You can also create the venv from VS Code itself: Ctrl+Shift+P → **Python: Create Environment** → **Venv** → pick Python 3.11 or 3.12. It makes the same `.venv` folder. You still need step 4 to install the packages.

## 4. Install the packages

Update pip first:

```bash
python -m pip install --upgrade pip
```

Install PyTorch **built for CUDA 12.8**. The plain `pip install torch` may install a build that doesn't support the 5080.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Install the rest. VS Code needs `ipykernel` to run notebooks in the venv.

```bash
pip install numpy scipy networkx matplotlib pygsp ipykernel
```

## 5. Check the GPU from Python

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
```

You should see:
- a torch version of 2.7 or newer,
- `12.8` (or higher),
- `True`,
- `NVIDIA GeForce RTX 5080`,
- `sm_120` somewhere in the list.

If `sm_120` is missing or the GPU isn't found, reinstall torch with the `cu128` command from step 4.

## 6. Open the notebook and select the venv

1. In the Explorer, open `Codes/SBM/notebooks/spectre_gpu_training.ipynb`.
2. Click **Select Kernel** at the top right of the notebook.
3. Choose **Python Environments…**, then the one marked **.venv** (its path ends in `.venv\Scripts\python.exe` on Windows or `.venv/bin/python` on Linux).
4. The kernel name at the top right should now show `.venv`.

If `.venv` isn't in the list: Ctrl+Shift+P → **Python: Select Interpreter** → **Enter interpreter path…** → choose `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux). Then pick the kernel again.

## 7. Run it

1. **Part B (config):** check the settings. The defaults train Planar for 100,000 steps with DeFoG's network. For a quick first try, set `total_steps=2_000` and `eval_every=1_000`.
2. Click **Run All** at the top of the notebook.
3. **Part A** checks the GPU, PyTorch, PyGSP and ORCA, and the data. Each line should say *found* or *available*. On Linux, ORCA is compiled automatically the first time.
4. **Part D** times 10 training steps and prints:
   - **the estimated total training time**
   - **the peak GPU memory.** If it runs out of memory, lower `batch_size` in Part B (for example 32), then click **Restart** and **Run All** again.
5. **Part F** trains. Progress lines appear under the cell. Every `eval_every` steps it scores the validation set at 1, 4 and 10 steps and saves a checkpoint.
6. **Parts G–I** plot the training curves, score the test set with DeFoG's protocol (5 runs × 40 graphs), and draw generated graphs.

**During long runs:**
- Keep VS Code open; closing the window stops the kernel.
- Keep the machine from sleeping (Windows: Settings → System → Power → Sleep: Never).
- Training output can get long. If VS Code shows *Output is truncated*, click it to open the full log in a text editor. Everything important is also saved to `history.json` (see step 9).

## 8. Stopping and resuming

- Training saves `last.pt` every `eval_every` steps.
- To stop, click the **Interrupt** button (■) at the top of the notebook.
- To continue, click **Run All** again. Part F prints `resumed from step ...` and carries on. This also works after closing VS Code or restarting the machine.
- To start a fresh run instead, set `resume=False` or change `run_name` in Part B.

## 9. Where the results go

Each run gets a folder `Codes/SBM/results_spectre/<run_name>/` containing:

| File | What it is |
|---|---|
| `last.pt`, `best.pt` | checkpoints. Large, and ignored by git |
| `history.json` | training loss and every validation score |
| `test_results.json` | the final test table |

To share results back, commit and push the two `.json` files (VS Code's Source Control panel works too):

```bash
git add Codes/SBM/results_spectre/*/history.json Codes/SBM/results_spectre/*/test_results.json
```

```bash
git commit -m "Planar GPU run results"
```

```bash
git push
```

## 10. If something goes wrong

| Problem | Fix |
|---|---|
| `WARNING: this PyTorch build has no kernels for this GPU` | Reinstall torch with the `cu128` command in step 4 |
| `CUDA out of memory` | Lower `batch_size` in Part B (64 → 32 → 16), then Restart and Run All |
| `ORCA build failed` (Linux) | `sudo apt install build-essential`, then rerun Part A. Or set `orbit=False` to skip the orbit metric |
| `pygsp missing` | `pip install pygsp` in the activated venv, then click **Restart** on the notebook |
| VS Code asks to install `ipykernel` | Click **Install**, or run `pip install ipykernel` in the activated venv |
| `.venv` is not in the kernel list | See the note at the end of step 6, then reload the window (Ctrl+Shift+P → **Developer: Reload Window**) |
| `StopIteration` in the first code cell, or `ModuleNotFoundError: tacfm_d` | Part A looks for `Codes/SBM` from the notebook's working folder. Keep the VS Code setting **Jupyter: Notebook File Root** at its default `${fileDirname}`, or set it to `${workspaceFolder}` with the repository root open |

## 11. SBM later

Set `dataset="sbm"` in Part B. The SBM validity check needs `graph_tool`, which can't be installed with pip. It's only available through conda on Linux or macOS (`conda install -c conda-forge graph-tool`). Without it, the notebook still trains and reports every MMD, and reports V.U.N. without the validity check.
