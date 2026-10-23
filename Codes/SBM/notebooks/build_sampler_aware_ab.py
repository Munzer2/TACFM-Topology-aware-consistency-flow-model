"""Builds Codes/SBM/notebooks/sampler_aware_ab.ipynb (unexecuted).

Run from anywhere:  python Codes/SBM/notebooks/build_sampler_aware_ab.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

cells = []

cells.append(md(r"""# Short A/B test: does sampler-aware training help 4-step generation?

**Question.** The trained Planar model keeps most of its *own* mistakes when the sampler re-noises its earlier guess (66–87% kept, versus 9–63% for random noise), so the later sampling steps barely improve on the first guess. Sampler-aware training trains it on exactly that kind of input. Does that fix it?

**Design.** Two short fine-tunes from the **same starting weights** (the finished `prev` run), identical in everything except the training input:

| Arm | Training |
|---|---|
| `control` | the normal TACFM-D loss |
| `sampler_aware` | half of each batch is built like the sampler does: real graph → noised to time s → **the model's own guess** → noised to time t → recover the real graph |

The control matters: fine-tuning alone (a fresh learning-rate schedule) could improve things, so only the **difference between the two arms** counts.

**Measured for the starting weights and both arms:**
1. **Self-correction check:** how many of its *own* wrong edges the model keeps (lower is better), and whether a step improves on the previous guess.
2. **Test set, DeFoG protocol:** 4- and 10-step scores, 5 runs × 40 graphs.

**Time on the RTX 5080:** about 35 min (control) + 45 min (sampler-aware) + 10–15 min of scoring. Each arm saves its progress, so rerunning the notebook continues where it stopped.
"""))

cells.append(md(r"""## A. Setup"""))

cells.append(code(r"""import sys, os, time, json, copy, math, pickle, subprocess, warnings
from pathlib import Path

import numpy as np
import networkx as nx
import torch
warnings.filterwarnings("ignore")

HERE = Path.cwd()
SBM_DIR = next(p for p in [HERE, HERE.parent, HERE / "Codes" / "SBM"] if (p / "tacfm_d.py").exists())
sys.path.insert(0, str(SBM_DIR))

from tacfm_d import (SlotLayout, graphs_to_slots, slots_to_graphs, edge_prior, batch_loss, generate,
                     sample_path, top_k_slots, GCNDiscrete)
from defog_nn import SlotGraphTransformer
from spectre_eval import mmd_metrics, ratios, vun_metrics, MMD_KEYS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    print("GPU:", torch.cuda.get_device_name(), "| torch", torch.__version__)
else:
    print("No GPU: CPU only (fine for a smoke test only)")

ORCA_DIR = SBM_DIR / "spectre_eval" / "orca"
ORCA_BIN = ORCA_DIR / ("orca.exe" if os.name == "nt" else "orca")
if not ORCA_BIN.exists():
    cmd = ["g++", "-O2", "-std=c++11", "-w", "-o", str(ORCA_BIN), str(ORCA_DIR / "orca.cpp")]
    subprocess.run(cmd[:1] + (["-static"] if os.name == "nt" else []) + cmd[1:], check=False)
print("ORCA:", "found" if ORCA_BIN.exists() else "MISSING (orbit MMD will fail)")
"""))

cells.append(md(r"""## B. Config

`init_from` is the starting point for both arms: the exported weights of the finished `prev` run (pushed to GitHub). Everything about the model and sampler (network size, ordering, prior, coupling, target, `topk` / `hard` / `poly3` sampling) is read from that run's config, so both arms continue exactly that setup.
"""))

cells.append(code(r"""CFG = dict(
    init_from="results_spectre/planar_transformer_bfs_slot_coup_prev_s0/export_last.pt",
    steps=5_000,               # fine-tuning steps per arm
    batch_size=64,
    lr=1e-4,                   # peak learning rate of the fine-tune (the base run peaked at 2e-4)
    warmup_steps=200,
    weight_decay=1e-12,
    grad_clip=1.0,
    amp=True,
    ema_target=0.995,          # consistency-target EMA (as in the base run)
    ema_sample=0.999,          # sampling EMA (shorter than the base run's 0.9995, to fit 5k steps)
    self_correct=0.5,          # sampler_aware arm: fraction of each batch built from the model's own guesses
    sc_start=(0.3, 0.95),      # time s of the guess
    sc_jump=(0.4, 0.9),        # t = s + (1 - s) * U[sc_jump]
    eval_steps=[4, 10],
    num_samples=40,
    final_runs=5,
    seed=0,
    out_dir="results_spectre/ab_sampler_aware",
)
ARMS = {"control": 0.0, "sampler_aware": CFG["self_correct"]}

init = torch.load(SBM_DIR / CFG["init_from"], map_location="cpu", weights_only=False)
BASE = init["cfg"]
OUT = SBM_DIR / CFG["out_dir"]
OUT.mkdir(parents=True, exist_ok=True)
print(f"starting weights: {CFG['init_from']}  (trained {init['step']:,} steps)")
print({k: BASE[k] for k in ("dataset", "ordering", "prior", "coupling", "target", "lam", "final", "renoise", "schedule")})
"""))

cells.append(md(r"""## C. Data and model"""))

cells.append(code(r"""SPECTRE_DIR = SBM_DIR / "data" / "spectre"
load = lambda part: pickle.load(open(SPECTRE_DIR / f"{BASE['dataset']}_{part}.pkl", "rb"))
train_g, val_g, test_g = load("train"), load("val"), load("test")
N_MAX = init["n_max"]
layout = SlotLayout(N_MAX, DEVICE)
x1_train, n_train = graphs_to_slots(train_g, layout, BASE["ordering"])
m_train = x1_train.sum(1).long()
p_slot = edge_prior(x1_train, layout.mask(n_train), BASE["prior"])
INIT_WEIGHTS = {k: (v.float() if v.is_floating_point() else v) for k, v in init["weights"].items()}

def build_model():
    if BASE["backbone"] == "transformer":
        m = SlotGraphTransformer(N_MAX, BASE["n_layers"], BASE["hidden_mlp_dims"], BASE["hidden_dims"],
                                 BASE["features"], BASE["rrwp_steps"], amp=CFG["amp"])
    else:
        m = GCNDiscrete(N_MAX)
    m.load_state_dict(INIT_WEIGHTS)
    return m.to(DEVICE)

def frozen_copy(model):
    c = copy.deepcopy(model).eval()
    for q in c.parameters():
        q.requires_grad = False
    return c

print(f"{BASE['dataset']}: train {len(train_g)} / test {len(test_g)} | N_MAX {N_MAX} | "
      f"parameters {sum(q.numel() for q in build_model().parameters()):,}")
"""))

cells.append(md(r"""## D. Measurements

- **`self_correction(model)`:** on 32 validation graphs × 4, builds the input two ways at the 4-step sampler's later jumps: **A** the real graph noised (what training shows) and **B** the model's own guess from time s, noised (what sampling shows). It reports the share of wrong input edges the model keeps, and whether its output beats the guess it started from (`gain`).
- **`test_table(model)`:** the DeFoG test protocol at 4 and 10 steps, with the base run's sampler settings.
"""))

cells.append(code(r"""X1v, Nv = graphs_to_slots(val_g, layout, BASE["ordering"])
X1v, Nv = X1v.repeat(4, 1), Nv.repeat(4)
MASKv = layout.mask(Nv); P0v = p_slot.unsqueeze(0) * MASKv; Kv = X1v.sum(1).long()
PAIRS = [(0.3, 0.58), (0.58, 0.88), (0.88, 0.98)]

@torch.no_grad()
def _guess(model, x, t):
    out = []
    for i in range(0, len(x), 32):
        xb = x[i:i + 32]
        out.append(torch.sigmoid(model(xb, torch.full((len(xb), 1), t, device=DEVICE), Nv[i:i + 32])))
    return top_k_slots(torch.cat(out) * MASKv, Kv, MASKv)

@torch.no_grad()
def self_correction(model, seed=0):
    model.eval()
    torch.manual_seed(seed)
    prec = lambda g: ((g * X1v).sum(1) / Kv).mean().item()
    def kept(x_in, pred):
        wrong = (x_in == 1) & (X1v == 0)
        return ((pred == 1) & wrong).sum().item() / max(1, wrong.sum().item())
    rows = {}
    for s, t in PAIRS:
        tt = lambda v: torch.full((len(X1v), 1), v, device=DEVICE)
        xA = sample_path(torch.rand_like(X1v), P0v, X1v, tt(t)); pA = _guess(model, xA, t)
        g = _guess(model, sample_path(torch.rand_like(X1v), P0v, X1v, tt(s)), s)
        xB = sample_path(torch.rand_like(X1v), P0v, g, tt(t)); pB = _guess(model, xB, t)
        rows[f"{s}->{t}"] = {"random_kept": kept(xA, pA), "own_kept": kept(xB, pB),
                             "guess_right": prec(g), "output_right": prec(pB), "gain": prec(pB) - prec(g)}
    return rows

REF_TEST = mmd_metrics(test_g, train_g)

@torch.no_grad()
def test_table(model):
    model.eval()
    table = {}
    for steps in CFG["eval_steps"]:
        runs = []
        for r in range(CFG["final_runs"]):
            torch.manual_seed(1000 + r)
            idx = torch.randint(len(n_train), (CFG["num_samples"],), device=DEVICE)
            n, m = n_train[idx], m_train[idx]
            x = generate(model, p_slot, n, layout, steps, final=BASE["final"], n_edges=m,
                         renoise=BASE.get("renoise", "hard"), schedule=BASE.get("schedule", "poly3"))
            gen = slots_to_graphs(x, n, layout)
            out = mmd_metrics(test_g, gen)
            out.update(ratios(out, REF_TEST))
            out.update(vun_metrics(gen, train_g, BASE["dataset"], BASE["dataset"] == "planar"))
            runs.append(out)
        table[steps] = {k: (float(np.mean([x[k] for x in runs])), float(np.std([x[k] for x in runs]))) for k in runs[0]}
    return table

def show_sc(rows, title):
    print(title)
    print(f"  {'jump':>10} {'random kept':>12} {'own kept':>9} {'guess right':>12} {'output right':>13} {'gain':>7}")
    for k, r in rows.items():
        print(f"  {k:>10} {r['random_kept']:12.1%} {r['own_kept']:9.1%} {r['guess_right']:12.1%} "
              f"{r['output_right']:13.1%} {r['gain']:+7.1%}")

def show_test(table, title):
    print(title)
    for s, r in table.items():
        print(f"  {s:2d}-step  " + "  ".join(f"{k}={r[k][0]:.4f}+/-{r[k][1]:.4f}" for k in
              ("degree", "clustering", "orbit", "spectre", "wavelet", "average_ratio", "valid", "vun")))
"""))

cells.append(md(r"""## E. Starting point (before any fine-tuning)"""))

cells.append(code(r"""RESULTS_PATH = OUT / "ab_results.json"
RESULTS = json.load(open(RESULTS_PATH)) if RESULTS_PATH.exists() else {}

def save_results():
    json.dump({"cfg": CFG, "base_cfg": BASE, "init_step": init["step"], **RESULTS}, open(RESULTS_PATH, "w"), indent=1)

if "start" not in RESULTS:
    m0 = build_model()
    RESULTS["start"] = {"self_correction": self_correction(m0), "test": test_table(m0)}
    save_results()
show_sc(RESULTS["start"]["self_correction"], "START: self-correction check")
show_test({int(k): v for k, v in RESULTS["start"]["test"].items()}, "START: test set")
"""))

cells.append(md(r"""## F. Fine-tune both arms

Same seed, same steps, same learning-rate schedule; the only difference is `self_correct`. Progress is saved every 500 steps and on interrupt (■); rerunning continues.
"""))

cells.append(code(r"""def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp"); torch.save(obj, tmp); os.replace(tmp, path)

@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, ps in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(ps.detach(), alpha=1 - decay)

def lr_at(step):
    if step < CFG["warmup_steps"]:
        return (step + 1) / CFG["warmup_steps"]
    prog = (step - CFG["warmup_steps"]) / max(1, CFG["steps"] - CFG["warmup_steps"])
    return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

def finetune(arm, self_correct):
    arm_dir = OUT / arm; arm_dir.mkdir(exist_ok=True)
    done_path, last_path = arm_dir / "done.pt", arm_dir / "last.pt"
    student = build_model(); ema_tgt = frozen_copy(student); ema_smp = frozen_copy(student)
    if done_path.exists():
        ema_smp.load_state_dict(torch.load(done_path, map_location=DEVICE, weights_only=False)["ema_smp"])
        print(f"[{arm}] already trained, loaded")
        return ema_smp
    opt = torch.optim.AdamW(student.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, losses = 0, []
    torch.manual_seed(CFG["seed"])
    if last_path.exists():
        ck = torch.load(last_path, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ck["student"]); ema_tgt.load_state_dict(ck["ema_tgt"]); ema_smp.load_state_dict(ck["ema_smp"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step, losses = ck["step"], ck["losses"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if ck["cuda_rng"] is not None and DEVICE.type == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in ck["cuda_rng"]])
        print(f"[{arm}] resumed from step {step:,}")
    save = lambda path: atomic_save({"student": student.state_dict(), "ema_tgt": ema_tgt.state_dict(),
                                     "ema_smp": ema_smp.state_dict(), "opt": opt.state_dict(),
                                     "sched": sched.state_dict(), "step": step, "losses": losses,
                                     "torch_rng": torch.get_rng_state(),
                                     "cuda_rng": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else None}, path)
    student.train()
    t0, run = time.time(), []
    try:
        while step < CFG["steps"]:
            idx = torch.randint(len(x1_train), (CFG["batch_size"],), device=DEVICE)
            loss = batch_loss(student, ema_tgt, x1_train[idx], n_train[idx], p_slot, layout,
                              coupling=BASE["coupling"], lam=BASE["lam"], target=BASE["target"],
                              self_correct=self_correct, sc_start=CFG["sc_start"], sc_jump=CFG["sc_jump"])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), CFG["grad_clip"])
            opt.step(); sched.step()
            ema_update(ema_tgt, student, CFG["ema_target"]); ema_update(ema_smp, student, CFG["ema_sample"])
            step += 1; run.append(loss.item())
            if step % 250 == 0:
                losses.append((step, float(np.mean(run))))
                print(f"[{arm}] step {step:5,}  loss {np.mean(run):.4f}  {len(run) / (time.time() - t0):.1f} it/s", flush=True)
                t0, run = time.time(), []
            if step % 500 == 0:
                save(last_path)
    except KeyboardInterrupt:
        save(last_path); print(f"[{arm}] interrupted: saved step {step:,}. Rerun to continue."); raise
    save(done_path)
    atomic_save({"cfg": {**BASE, **{"finetune": CFG, "arm": arm}}, "n_max": N_MAX, "step": init["step"] + step,
                 "best_step": None, "val": None,
                 "weights": {k: (v.half() if v.is_floating_point() else v) for k, v in ema_smp.state_dict().items()}},
                arm_dir / f"export_{arm}.pt")
    return ema_smp

MODELS = {arm: finetune(arm, sc) for arm, sc in ARMS.items()}
"""))

cells.append(md(r"""## G. Compare"""))

cells.append(code(r"""for arm, model in MODELS.items():
    if arm not in RESULTS:
        RESULTS[arm] = {"self_correction": self_correction(model), "test": test_table(model)}
        save_results()

for name in ["start", *ARMS]:
    show_sc(RESULTS[name]["self_correction"], f"\n{name.upper()}: self-correction check")
    show_test({int(k): v for k, v in RESULTS[name]["test"].items()}, f"{name.upper()}: test set")

def g(name, steps, key):
    r = RESULTS[name]["test"]; r = r.get(steps, r.get(str(steps)))
    return r[key]
print("\nSUMMARY (lower ratio is better; higher V.U.N. is better; lower 'own kept' is better)")
head = "".join(f"{f'{st}-step ratio':>18}{f'{st}-step VUN':>13}" for st in CFG["eval_steps"])
print(f"{'':>15}{head}{'own kept .58->.88':>19}{'own kept .88->.98':>19}")
for name in ["start", *ARMS]:
    sc = RESULTS[name]["self_correction"]
    cols = "".join(f"{g(name, st, 'average_ratio')[0]:11.1f}+/-{g(name, st, 'average_ratio')[1]:<4.1f}"
                   f"{g(name, st, 'vun')[0]:13.1%}" for st in CFG["eval_steps"])
    print(f"{name:>15}{cols}{sc['0.58->0.88']['own_kept']:19.1%}{sc['0.88->0.98']['own_kept']:19.1%}")
"""))

cells.append(md(r"""## H. How to read it, and what to push

**Sampler-aware training works if, compared with `control` (not with `start`):**
1. **`own kept` drops clearly** (e.g. from ~70% toward the ~10% the model keeps of random noise), and `gain` becomes positive (a step now improves on the guess), **and**
2. the **4-step ratio is lower** and/or **4-step V.U.N. is higher**, by more than the `+/-` spread.

If (1) holds but (2) doesn't, the model learned to fix its mistakes but 5k steps weren't enough to show in the samples: a longer run is the next step. If neither holds, the idea doesn't help here.

**Push the results** (small files) from the repository root:

```bash
git add Codes/SBM/results_spectre/ab_sampler_aware/ab_results.json Codes/SBM/results_spectre/ab_sampler_aware/*/export_*.pt
```

```bash
git commit -m "Sampler-aware A/B test results"
```

```bash
git push
```
"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).with_name("sampler_aware_ab.ipynb")
nbformat.write(nb, out)
print("wrote", out)
