"""Builds Codes/SBM/notebooks/spectre_gpu_training.ipynb (unexecuted).

Run from anywhere:  python Codes/SBM/notebooks/build_gpu_training.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

cells = []

# ======================================================================
cells.append(md(r"""# TACFM-D on SPECTRE (Planar / SBM): GPU training

This notebook trains TACFM-D with **DeFoG's graph transformer** as the network and scores it with the **DiGress/DeFoG evaluation code**, so the numbers are directly comparable with those papers.

**What is ours and what is borrowed**

| Part | Source |
|---|---|
| Path `x_t = 1[u < (1-t)·p0 + t·x1]`, per-slot prior, consistency loss, degree-sequence coupling, 1–10 step sampler | **ours** (`tacfm_d.py`) |
| Graph transformer + RRWP / cycle features | DeFoG (MIT), vendored unchanged in `defog_nn/` |
| Metrics: degree / clustering / orbit / spectral / wavelet MMD, ratios, V.U.N. | DiGress/DeFoG (MIT), vendored in `spectre_eval/` |
| Data and train/val/test split (128/32/40) | SPECTRE files, split exactly as DiGress/DeFoG |

**How to run:** open in Jupyter/VS Code from `Codes/SBM/notebooks/`, set the config in Part B, and run all cells. Training saves a checkpoint every `eval_every` steps and **resumes automatically** if you rerun it, so it is safe to stop and restart.

| Part | What it does |
|---|---|
| A | Environment check (GPU, PyTorch, ORCA, PyGSP, data) |
| B | Config |
| C | Data |
| D | Model, speed and memory check |
| E | Evaluation helpers |
| F | **Training** (resumable) |
| G | Training curves |
| H | **Final test evaluation** (DeFoG protocol: 5 runs × 40 graphs) |
| I | Look at generated graphs |
| J | Notes for SBM |
"""))

# ======================================================================
cells.append(md(r"""## A. Environment

The RTX 5080 (Blackwell, `sm_120`) needs **PyTorch ≥ 2.7 built for CUDA 12.8** (`cu128` wheels):

```
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install networkx scipy matplotlib pygsp
```
"""))

cells.append(code(r"""import sys, os, platform, subprocess, time, json, copy, math, pickle, shutil, warnings
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Find Codes/SBM whether this runs from notebooks/ or from SBM/.
HERE = Path.cwd()
SBM_DIR = next(p for p in [HERE, HERE.parent, HERE / "Codes" / "SBM"] if (p / "tacfm_d.py").exists())
sys.path.insert(0, str(SBM_DIR))

print(f"python {platform.python_version()} | torch {torch.__version__} | networkx {nx.__version__}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    cap = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} | compute capability {cap} | "
          f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB | CUDA {torch.version.cuda}")
    if cap >= (12, 0) and f"sm_{cap[0]}{cap[1]}" not in torch.cuda.get_arch_list():
        print("WARNING: this PyTorch build has no kernels for this GPU. Install a cu128 build (see above).")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("No GPU found: running on CPU (fine for a smoke test, far too slow for real training).")
print("code folder:", SBM_DIR)
"""))

cells.append(code(r"""# --- PyGSP (wavelet metric) ---
try:
    import pygsp
    print("pygsp", pygsp.__version__)
except ImportError:
    print("pygsp missing: run  %pip install pygsp  (needed for the wavelet MMD)")

# --- ORCA (orbit metric): build it from source if the binary is missing ---
ORCA_DIR = SBM_DIR / "spectre_eval" / "orca"
ORCA_BIN = ORCA_DIR / ("orca.exe" if os.name == "nt" else "orca")
if not ORCA_BIN.exists():
    cmd = ["g++", "-O2", "-std=c++11", "-w", "-o", str(ORCA_BIN), str(ORCA_DIR / "orca.cpp")]
    if os.name == "nt":
        cmd.insert(1, "-static")
    print("building ORCA:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        print("ORCA build failed (", e, "). Install g++ (Linux: build-essential; Windows: MSYS2 mingw-w64), "
              "or set ORBIT=False in the config to skip the orbit metric.")
print("ORCA binary:", ORCA_BIN, "found" if ORCA_BIN.exists() else "MISSING")

# --- graph_tool (only for SBM validity; Linux/macOS: conda install -c conda-forge graph-tool) ---
try:
    import graph_tool  # noqa: F401
    HAS_GRAPH_TOOL = True
except ImportError:
    HAS_GRAPH_TOOL = False
print("graph_tool:", "available" if HAS_GRAPH_TOOL else "not installed (only needed for SBM validity)")
"""))

cells.append(code(r"""# --- SPECTRE data with the DiGress/DeFoG split ---
SPECTRE_DIR = SBM_DIR / "data" / "spectre"
RAW_URLS = {"planar": "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt",
            "sbm": "https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt"}
if not all((SPECTRE_DIR / f"{d}_{s}.pkl").exists() for d in RAW_URLS for s in ("train", "val", "test")):
    import urllib.request
    (SPECTRE_DIR / "raw").mkdir(parents=True, exist_ok=True)
    for d, url in RAW_URLS.items():
        dst = SPECTRE_DIR / "raw" / url.rsplit("/", 1)[1]
        if not dst.exists():
            print("downloading", url)
            urllib.request.urlretrieve(url, dst)
    subprocess.run([sys.executable, "-I", str(SBM_DIR / "data" / "prepare_spectre.py")], check=True)
print(sorted(p.name for p in SPECTRE_DIR.glob("*.pkl")))
"""))

# ======================================================================
cells.append(md(r"""## B. Config

Defaults reproduce DeFoG's Planar setup for everything that is not part of our method (network size, RRWP features, batch 64, AdamW lr 2e-4).

- `target`: `"prev"` is the original TACFM-D (EMA target at the noisier time `t−δ`); `"next"` follows the original consistency-model recipe (target at `t+δ`, closer to the data).
- `self_cond`: self-conditioning, i.e. the model also gets its previous prediction as an extra input (empty at the first sampling step). Test it with `sampler_aware_ab.ipynb` before training from scratch. The run folder gets `_sc` in its name.
- `alpha`: the share of the loss given to "match the real graph" (the anchor). `None` uses `lam` the old way (`lam=3` = a 75% share). The literature (CDLM) uses 10–40%; test with `sampler_aware_ab.ipynb` first.
- `final`: how the last prediction becomes edges. `"topk"` keeps the `m` most likely edges, with `m` taken from a training graph (fixes empty 1-step graphs on sparse data); `"threshold"` is the old `p > 0.5`; `"sample"` draws each edge.
- `renoise`: between sampling steps, `"hard"` re-noises **one concrete graph** (the prediction turned into edges with the `final` rule), which matches training; `"soft"` re-noises the predicted probabilities (the original sampler). On Planar, `"hard"` is much better.
- `schedule`: `"poly3"` packs the sampling steps toward t = 1, where the last few wrong edges are fixed; `"uniform"` spaces them evenly.
- `sample_with`: generate with a slow EMA of the weights (`"ema_sample"`) or the raw student.
- `total_steps`: DeFoG trains Planar for ~200k optimizer steps (100k epochs × 2 batches). 100k is a good first run.
"""))

cells.append(code(r"""from defog_nn import PRESETS

CFG = dict(
    dataset="planar",          # "planar" or "sbm"
    backbone="transformer",    # "transformer" (DeFoG) or "gcn" (our small GCN, for the ablation)
    ordering="bfs",            # node ordering before padding: "bfs", "degree", "none"
    prior="slot",              # "slot" (one edge probability per slot) or "global"
    coupling=True,             # topology-aware (degree-sequence) coupling
    target="prev",             # "prev" or "next" (see above)
    lam=3.0,                   # weight of the BCE-to-real-graph term (used only when alpha is None)
    alpha=None,                # anchor share of the loss, 0..1: 4*((1-alpha)*consistency + alpha*anchor);
                               # 0.75 equals lam=3 exactly. None keeps consistency + lam * anchor
    self_cond=False,           # self-conditioning: feed the previous prediction back as an extra edge input
    ema_target=0.995,          # EMA used as the consistency target
    ema_sample=0.9995,         # slower EMA used for sampling
    sample_with="ema_sample",  # "ema_sample" or "student"
    batch_size=64,
    lr=2e-4,
    weight_decay=1e-12,
    warmup_steps=1000,
    lr_schedule="cosine",      # "cosine" or "constant"
    grad_clip=1.0,
    amp=True,                  # bfloat16 autocast for the transformer on GPU
    total_steps=100_000,
    eval_every=5_000,          # validation scoring, best.pt, small weight exports
    ckpt_every=500,            # how often last.pt is saved (also saved when you interrupt)
    log_every=200,
    eval_steps=[1, 4, 10],     # sampling step counts to score
    select_steps=4,            # pick the best checkpoint by validation average ratio at this step count
    final="topk",              # "topk", "threshold", "sample"
    renoise="hard",            # between steps, re-noise one concrete graph ("hard") or the probabilities ("soft")
    schedule="poly3",          # step times: "uniform" or "polyP" (packed toward t = 1)
    num_samples=40,            # graphs per evaluation (DeFoG: 40)
    final_runs=5,              # test runs for the final table (DeFoG: 5)
    orbit=True,                # orbit MMD needs the ORCA binary
    seed=0,
    run_name=None,             # default: derived from the settings
    resume=True,
)
CFG.update(PRESETS[CFG["dataset"]])     # n_layers, features, rrwp_steps, hidden dims (DeFoG's values)
if CFG["dataset"] == "sbm":
    CFG["batch_size"] = 32               # DeFoG's SBM batch size

CFG["run_name"] = CFG["run_name"] or (f"{CFG['dataset']}_{CFG['backbone']}_{CFG['ordering']}_{CFG['prior']}_"
                                      f"{'coup' if CFG['coupling'] else 'indep'}_{CFG['target']}"
                                      f"{'_sc' if CFG['self_cond'] else ''}_s{CFG['seed']}")
RUN_DIR = SBM_DIR / "results_spectre" / CFG["run_name"]
RUN_DIR.mkdir(parents=True, exist_ok=True)
print("run folder:", RUN_DIR)
print(json.dumps(CFG, indent=1))
"""))

# ======================================================================
cells.append(md(r"""## C. Data

Graphs are reordered (BFS from the highest-degree node by default), zero-padded to `N_MAX`, and flattened to the upper-triangle slot vector. `n_train` / `m_train` are the node and edge counts of the training graphs; generation draws `(n, m)` pairs from them.
"""))

cells.append(code(r"""from tacfm_d import (SlotLayout, graphs_to_slots, slots_to_graphs, edge_prior, batch_loss, generate,
                     GCNDiscrete)

torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])

def load_split(part):
    return pickle.load(open(SPECTRE_DIR / f"{CFG['dataset']}_{part}.pkl", "rb"))

train_g, val_g, test_g = load_split("train"), load_split("val"), load_split("test")
N_MAX = max(G.number_of_nodes() for G in train_g + val_g + test_g)
layout = SlotLayout(N_MAX, DEVICE)
x1_train, n_train = graphs_to_slots(train_g, layout, CFG["ordering"])
m_train = x1_train.sum(1).long()
p_slot = edge_prior(x1_train, layout.mask(n_train), CFG["prior"])

print(f"{CFG['dataset']}: train/val/test {len(train_g)}/{len(val_g)}/{len(test_g)} | N_MAX {N_MAX} | slots {layout.d}")
print(f"nodes {n_train.min().item()}-{n_train.max().item()} | edges {m_train.min().item()}-{m_train.max().item()} "
      f"(mean {m_train.float().mean():.1f})")

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, G in zip(axes, train_g[:4]):
    pos = nx.planar_layout(G) if CFG["dataset"] == "planar" else nx.spring_layout(G, seed=0)
    nx.draw(G, pos=pos, ax=ax, node_size=12, edge_color="gray")
    ax.set_title(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
plt.suptitle("training graphs"); plt.tight_layout(); plt.show()
"""))

# ======================================================================
cells.append(md(r"""## D. Model, speed and memory

`SlotGraphTransformer` wraps DeFoG's network: it builds the one-hot edge states and extra features from `x_t`, puts `t` in the global features, and returns one logit per edge slot. The second cell times a few training steps and estimates the total run time. If you run out of GPU memory, lower `batch_size`.
"""))

cells.append(code(r"""from defog_nn import SlotGraphTransformer

def build_model():
    if CFG["backbone"] == "transformer":
        return SlotGraphTransformer(N_MAX, CFG["n_layers"], CFG["hidden_mlp_dims"], CFG["hidden_dims"],
                                    CFG["features"], CFG["rrwp_steps"], amp=CFG["amp"],
                                    self_cond=CFG["self_cond"]).to(DEVICE)
    assert not CFG["self_cond"], "self-conditioning needs the transformer backbone"
    return GCNDiscrete(N_MAX).to(DEVICE)

def frozen_copy(model):
    c = copy.deepcopy(model).eval()
    for q in c.parameters():
        q.requires_grad = False
    return c

student = build_model()
ema_tgt = frozen_copy(student)       # consistency target
ema_smp = frozen_copy(student)       # sampling weights
opt = torch.optim.AdamW(student.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

def lr_at(step):
    if step < CFG["warmup_steps"]:
        return (step + 1) / CFG["warmup_steps"]
    if CFG["lr_schedule"] == "constant":
        return 1.0
    prog = (step - CFG["warmup_steps"]) / max(1, CFG["total_steps"] - CFG["warmup_steps"])
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

print(f"{CFG['backbone']} parameters: {sum(q.numel() for q in student.parameters()):,}")
"""))

cells.append(code(r"""@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, ps in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(ps.detach(), alpha=1 - decay)

def train_step():
    idx = torch.randint(len(x1_train), (CFG["batch_size"],), device=DEVICE)
    loss = batch_loss(student, ema_tgt, x1_train[idx], n_train[idx], p_slot, layout,
                      coupling=CFG["coupling"], lam=CFG["lam"], target=CFG["target"], alpha=CFG["alpha"],
                      self_cond=CFG["self_cond"])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if CFG["grad_clip"]:
        torch.nn.utils.clip_grad_norm_(student.parameters(), CFG["grad_clip"])
    opt.step()
    sched.step()
    ema_update(ema_tgt, student, CFG["ema_target"])
    ema_update(ema_smp, student, CFG["ema_sample"])
    return loss.item()

# Speed / memory check on throw-away copies, so the real run starts from a clean state.
_saved = [copy.deepcopy(m.state_dict()) for m in (student, ema_tgt, ema_smp)] + [copy.deepcopy(opt.state_dict()),
                                                                                  copy.deepcopy(sched.state_dict())]
if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
n_bench = 10 if DEVICE.type == "cuda" else 2
train_step()
if DEVICE.type == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
for _ in range(n_bench):
    train_step()
if DEVICE.type == "cuda":
    torch.cuda.synchronize()
sec = (time.time() - t0) / n_bench
for m, sd in zip((student, ema_tgt, ema_smp), _saved[:3]):
    m.load_state_dict(sd)
opt.load_state_dict(_saved[3]); sched.load_state_dict(_saved[4])
print(f"{sec:.3f} s per training step (batch {CFG['batch_size']}) -> {CFG['total_steps']:,} steps ~ "
      f"{sec * CFG['total_steps'] / 3600:.1f} h (+ evaluation)")
if DEVICE.type == "cuda":
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
"""))

# ======================================================================
cells.append(md(r"""## E. Evaluation helpers

- **MMDs** (degree, clustering, orbit, spectral, wavelet) use the DiGress/DeFoG code unchanged (Gaussian total-variation kernel).
- **Ratio** = generated MMD ÷ the MMD between the training set and the reference set, averaged over the five metrics (DeFoG's "Ratio" column; 1.0 means as close as the training data itself).
- **V.U.N.** = fraction of generated graphs that are valid, unique and not in the training set.

During training we score against the 32 validation graphs; the final table uses the 40 test graphs.
"""))

cells.append(code(r"""from spectre_eval import mmd_metrics, ratios, vun_metrics, MMD_KEYS

KEYS = MMD_KEYS if CFG["orbit"] else tuple(k for k in MMD_KEYS if k != "orbit")
CHECK_VALIDITY = CFG["dataset"] == "planar" or HAS_GRAPH_TOOL

REF = {"val": mmd_metrics(val_g, train_g, KEYS), "test": mmd_metrics(test_g, train_g, KEYS)}
print("reference MMD, train vs val: ", {k: round(v, 5) for k, v in REF["val"].items()})
print("reference MMD, train vs test:", {k: round(v, 5) for k, v in REF["test"].items()})

def sampler_model():
    return ema_smp if CFG["sample_with"] == "ema_sample" else student

def sample_graphs(model, num, steps, final=None, times=None):
    idx = torch.randint(len(n_train), (num,), device=DEVICE)
    n, m = n_train[idx], m_train[idx]
    x = generate(model, p_slot, n, layout, steps, final=final or CFG["final"], n_edges=m, times=times,
                 renoise=CFG.get("renoise", "soft"), schedule=CFG.get("schedule", "uniform"),
                 self_cond=CFG["self_cond"])
    return slots_to_graphs(x, n, layout)

def score(gen, split):
    ref_graphs = val_g if split == "val" else test_g
    out = mmd_metrics(ref_graphs, gen, KEYS)
    out.update(ratios(out, REF[split]))
    out.update(vun_metrics(gen, train_g, CFG["dataset"], CHECK_VALIDITY))
    return out

SHORT = [("degree", "deg"), ("clustering", "clus"), ("orbit", "orb"), ("spectre", "spec"), ("wavelet", "wav"),
         ("average_ratio", "ratio"), ("valid", "valid"), ("vun", "VUN")]
def fmt(r):
    return "  ".join(f"{s}={r[k]:.4f}" for k, s in SHORT if k in r)
"""))

# ======================================================================
cells.append(md(r"""## F. Training

Rerunning this cell continues from `last.pt` in the run folder (set `CFG["resume"] = False` to start over).

- **Pausing:** click Interrupt (■). The cell catches it and saves `last.pt` before stopping, so no steps are lost. Rerun the cell (or Run All) to continue. `last.pt` is also saved every `ckpt_every` steps, in case the machine crashes.
- **Every `eval_every` steps:** scores the validation set at each step count in `eval_steps`, keeps `best.pt` (lowest validation average ratio at `select_steps`), and writes two **small weight files for GitHub**:
  - `export_last.pt`: the sampling weights in fp16 (about 18 MB), plus the config and validation scores
  - `export_best.pt`: the same, for the best checkpoint so far

  Unlike `last.pt` and `best.pt` (about 180 MB each, including the optimizer), these are small enough to push. `eval_spectre_export.py` scores them on a CPU.
"""))

cells.append(code(r"""def atomic_save(obj, path):
    # Write to a temporary file, then swap it in, so an interrupt never leaves a half-written checkpoint.
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def save_ckpt(path, state):
    atomic_save({"student": student.state_dict(), "ema_tgt": ema_tgt.state_dict(), "ema_smp": ema_smp.state_dict(),
                 "opt": opt.state_dict(), "sched": sched.state_dict(), "cfg": CFG, "state": state,
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None}, path)

def export_weights(path, state, val=None):
    # Sampling weights only, in fp16: small enough for GitHub (about 18 MB for the Planar transformer).
    w = {k: (v.half() if v.is_floating_point() else v) for k, v in sampler_model().state_dict().items()}
    atomic_save({"cfg": CFG, "n_max": N_MAX, "step": state["step"], "best_step": state["best_step"],
                 "val": val, "weights": w}, path)

def save_history(state):
    json.dump({"cfg": CFG, "state": state}, open(RUN_DIR / "history.json", "w"), indent=1)

state = {"step": 0, "history": [], "best_ratio": float("inf"), "best_step": None}
if CFG["resume"] and (RUN_DIR / "last.pt").exists():
    ck = torch.load(RUN_DIR / "last.pt", map_location=DEVICE, weights_only=False)
    student.load_state_dict(ck["student"]); ema_tgt.load_state_dict(ck["ema_tgt"]); ema_smp.load_state_dict(ck["ema_smp"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); state = ck["state"]
    # map_location moved the saved RNG states to the GPU; the RNG setters need them on the CPU.
    torch.set_rng_state(ck["torch_rng"].cpu())
    if ck["cuda_rng"] is not None and DEVICE.type == "cuda":
        torch.cuda.set_rng_state_all([r.cpu() for r in ck["cuda_rng"]])
    print(f"resumed from step {state['step']:,}")

t0, run_loss, n_loss = time.time(), 0.0, 0
try:
    while state["step"] < CFG["total_steps"]:
        run_loss += train_step(); n_loss += 1
        state["step"] += 1
        step = state["step"]

        if step % CFG["log_every"] == 0:
            el = time.time() - t0
            print(f"step {step:7,}  loss {run_loss / n_loss:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{n_loss / el:.1f} it/s", flush=True)
            state["history"].append({"step": step, "loss": run_loss / n_loss})
            t0, run_loss, n_loss = time.time(), 0.0, 0

        if step % CFG["eval_every"] == 0 or step == CFG["total_steps"]:
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None)
            entry = {"step": step, "val": {}}
            for s in CFG["eval_steps"]:
                entry["val"][s] = score(sample_graphs(sampler_model(), CFG["num_samples"], s), "val")
                print(f"  [val] step {step:7,}  {s:2d}-step  {fmt(entry['val'][s])}", flush=True)
            torch.set_rng_state(rng[0])
            if rng[1] is not None:
                torch.cuda.set_rng_state_all(rng[1])
            state["history"].append(entry)
            sel = entry["val"][CFG["select_steps"]]["average_ratio"]
            if sel < state["best_ratio"]:
                state["best_ratio"], state["best_step"] = sel, step
                save_ckpt(RUN_DIR / "best.pt", state)
                export_weights(RUN_DIR / "export_best.pt", state, entry["val"])
                print(f"  new best (ratio {sel:.2f} at {CFG['select_steps']} steps)")
            export_weights(RUN_DIR / "export_last.pt", state, entry["val"])
            save_ckpt(RUN_DIR / "last.pt", state)
            save_history(state)
            t0, run_loss, n_loss = time.time(), 0.0, 0
        elif step % CFG["ckpt_every"] == 0:
            save_ckpt(RUN_DIR / "last.pt", state)
            save_history(state)
except KeyboardInterrupt:
    save_ckpt(RUN_DIR / "last.pt", state)
    save_history(state)
    print(f"interrupted: saved last.pt at step {state['step']:,}. Rerun this cell to continue.")
    raise

print(f"done: {state['step']:,} steps, best validation ratio {state['best_ratio']:.2f} at step {state['best_step']}")
"""))

# ======================================================================
cells.append(md(r"""## G. Training curves"""))

cells.append(code(r"""hist = state["history"]
loss_pts = [(h["step"], h["loss"]) for h in hist if "loss" in h]
val_pts = [h for h in hist if "val" in h]

fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
if loss_pts:
    axes[0].plot(*zip(*loss_pts)); axes[0].set_title("training loss"); axes[0].set_xlabel("step")
for s in CFG["eval_steps"]:
    pts = [(h["step"], h["val"][s]["average_ratio"]) for h in val_pts if s in h["val"]]
    if pts:
        axes[1].plot(*zip(*pts), "o-", label=f"{s}-step")
    pts = [(h["step"], h["val"][s].get("vun", float("nan"))) for h in val_pts if s in h["val"]]
    if pts:
        axes[2].plot(*zip(*pts), "o-", label=f"{s}-step")
axes[1].set_yscale("log"); axes[1].set_title("validation average ratio (lower is better)"); axes[1].legend()
axes[2].set_title("validation V.U.N. (higher is better)"); axes[2].legend()
plt.tight_layout(); plt.show()
"""))

# ======================================================================
cells.append(md(r"""## H. Final test evaluation

Loads `best.pt` and scores the **test** set with DeFoG's protocol: for each step count, 5 runs of 40 generated graphs, mean ± std. It also compares the three ways of turning the last prediction into edges at `select_steps`. Results go to `test_results.json` in the run folder.
"""))

cells.append(code(r"""ck = torch.load(RUN_DIR / "best.pt", map_location=DEVICE, weights_only=False)
student.load_state_dict(ck["student"]); ema_smp.load_state_dict(ck["ema_smp"])
print(f"best checkpoint: step {ck['state']['best_step']:,}")

def test_table(steps_list, final=None, runs=CFG["final_runs"]):
    out = {}
    for s in steps_list:
        rs = []
        for r in range(runs):
            torch.manual_seed(1000 + r)
            rs.append(score(sample_graphs(sampler_model(), CFG["num_samples"], s, final=final), "test"))
        out[s] = {k: (float(np.mean([x[k] for x in rs])), float(np.std([x[k] for x in rs]))) for k in rs[0]}
    return out

def show(table, title):
    cols = [k for k, _ in SHORT if k in next(iter(table.values()))]
    print(title)
    print(f"{'steps':>6}" + "".join(f"{c:>20}" for c in cols))
    for s, row in table.items():
        print(f"{s:>6}" + "".join(f"{row[c][0]:>11.4f} +/-{row[c][1]:<7.4f}" for c in cols))

TEST = {"main": test_table(CFG["eval_steps"])}
show(TEST["main"], f"\nTEST, final='{CFG['final']}'")

TEST["final_modes"] = {f: test_table([CFG["select_steps"]], final=f) for f in ("topk", "threshold", "sample")}
for f, tab in TEST["final_modes"].items():
    show(tab, f"\nTEST at {CFG['select_steps']} steps, final='{f}'")

json.dump({"cfg": CFG, "best_step": ck["state"]["best_step"], "reference": REF,
           "test": {k: {str(s): v for s, v in t.items()} if k == "main" else
                    {f: {str(s): v for s, v in tab.items()} for f, tab in t.items()} for k, t in TEST.items()}},
          open(RUN_DIR / "test_results.json", "w"), indent=1)
print("\nwrote", RUN_DIR / "test_results.json")
"""))

# ======================================================================
cells.append(md(r"""## I. Generated graphs"""))

cells.append(code(r"""torch.manual_seed(0)
gen = sample_graphs(sampler_model(), 8, CFG["select_steps"])
valid = [(nx.is_connected(G) and nx.check_planarity(G)[0]) if CFG["dataset"] == "planar" else None for G in gen]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, G, ok in zip(axes.ravel(), gen, valid):
    is_planar, _ = nx.check_planarity(G)
    pos = nx.planar_layout(G) if is_planar else nx.spring_layout(G, seed=0)
    nx.draw(G, pos=pos, ax=ax, node_size=12, edge_color="gray")
    ax.set_title(f"{G.number_of_edges()} edges" + ("" if ok is None else ("  valid" if ok else "  NOT valid")))
plt.suptitle(f"{CFG['select_steps']}-step samples"); plt.tight_layout(); plt.show()
"""))

# ======================================================================
cells.append(md(r"""## J. Notes for SBM and for comparisons

- **SBM:** set `dataset="sbm"` in Part B (the preset switches to DeFoG's SBM network: 8 layers, 20 RRWP steps, batch 32). Graphs have 44–187 nodes, so memory and time per step are much higher. Validity needs `graph_tool` (Linux/macOS: `conda install -c conda-forge graph-tool`); without it, V.U.N. is reported without the validity check.
- **Ablations:** `backbone="gcn"` (our small GCN), `prior="global"`, `coupling=False`, `target="next"`, `ordering="degree"` / `"none"`. Each setting gets its own run folder.
- **Baselines at few steps:** the fair comparison is DeFoG itself sampled with 1, 4 and 10 steps on the same split and the same evaluation code.
- **Reporting:** DeFoG reports mean ± std over 5 sampling runs; add more training seeds (`seed=1, 2`) for the paper.
"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).with_name("spectre_gpu_training.ipynb")
nbformat.write(nb, out)
print("wrote", out)
