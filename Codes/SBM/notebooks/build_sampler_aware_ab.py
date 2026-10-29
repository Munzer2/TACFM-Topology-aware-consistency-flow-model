"""Builds Codes/SBM/notebooks/sampler_aware_ab.ipynb (unexecuted).

Run from anywhere:  python Codes/SBM/notebooks/build_sampler_aware_ab.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

cells = []

cells.append(md(r"""# Short A/B test: how much weight should the anchor get?

**Question.** The loss has two parts: the **consistency** term (match the EMA model's prediction from a cleaner point on the same path) and the **anchor** (match the real graph). Every run so far gave the anchor **75%** of the loss (α = 0.75, the old λ = 3). The literature uses 10–40% (CDLM) up to about 75% (Categorical Flow Maps). And **α = 1.0 switches the consistency term off**, which shows whether it helps at all.

**Design.** Four 5k-step runs, all with `target="next"`, differing only in α. With `from_scratch=True` (the default now) every arm starts from the **same random initialisation** (same seed) and uses the base run's learning rate and warmup. With `from_scratch=False` every arm is a fine-tune from the finished `prev` run instead (the earlier test, 2026-10-29: all α equal).

5k steps from scratch is early: expect low V.U.N. for every arm, so compare the MMDs, the average ratio and the training loss as well as V.U.N.

| Arm | α (anchor share) | What it tells us |
|---|---|---|
| `next_alpha_0.75` | 0.75 | today's setting: the baseline (Session 1 / self-conditioning test: 20–25% V.U.N. at 10 steps) |
| `next_alpha_0.5` | 0.5 | does more consistency help? |
| `next_alpha_0.3` | 0.3 | CDLM's range |
| `next_alpha_1.0` | 1.0 | **consistency off**: a plain denoiser. If this matches the others, the consistency term isn't doing anything |

With α, the loss is `4 × ((1 − α) × consistency + α × anchor)`, so every arm has the same total weight and α = 0.75 is exactly the old loss. Compare every arm with **`next_alpha_0.75`**.

**Measured for the starting weights and every arm:**
1. **Test set, DeFoG protocol:** scores at **4, 10 and 20 sampling steps**, 5 runs × 40 graphs each.
2. **Self-correction check:** how many of its own wrong edges the model keeps when its earlier guess is re-noised.

**Time on the RTX 5080:** about 40 min of training plus about 5 min of scoring per arm, so **about 3 hours**. Each arm saves its progress and finished arms are skipped, so you can stop at any time and rerun to continue. To run fewer arms, delete entries from `ARMS` in Part B.
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

from tacfm_d import (SlotLayout, graphs_to_slots, slots_to_graphs, edge_prior, batch_loss, generate, sampler_pairs,
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

`init_from` is the starting point for every arm (the final `prev` export): the exported weights of the finished `prev` run (pushed to GitHub). Everything about the model and sampler (network size, ordering, prior, coupling, target, `topk` / `hard` / `poly3` sampling) is read from that run's config, so both arms continue exactly that setup.
"""))

cells.append(code(r"""CFG = dict(
    init_from="results_spectre/planar_transformer_bfs_slot_coup_prev_s0/export_last.pt",  # config (and weights unless from_scratch)
    from_scratch=True,         # True: random init (same seed for every arm); False: fine-tune from init_from's weights
    steps=5_000,               # fine-tuning steps per arm
    batch_size=64,
    lr=1e-4,                   # peak learning rate of a fine-tune (the base run peaked at 2e-4)
    warmup_steps=200,
    scratch_lr=2e-4,           # from scratch: the base run's peak learning rate ...
    scratch_warmup_steps=1000, # ... and warmup
    weight_decay=1e-12,
    grad_clip=1.0,
    amp=True,
    ema_target=0.995,          # consistency-target EMA (as in the base run)
    ema_sample=0.999,          # sampling EMA (shorter than the base run's 0.9995, to fit 5k steps)
    sc_prob=0.5,               # self-conditioning: share of rows that get their own first prediction as the hint
    delta_range=(0.1, 0.3),    # gap between the student's time and the EMA target's time (the original range)
    same_u=True,               # sampler: re-noise with the same u at every step (better than fresh noise)
    eval_steps=[4, 10, 20],    # sampling step counts scored on the test set
    num_samples=40,
    final_runs=5,
    seed=0,
    out_dir="results_spectre/ab_alpha_scratch",
)
# Each arm = overrides of the base run's training settings. Delete entries to run fewer arms.
ARMS = {
    "next_alpha_0.75": {"target": "next", "alpha": 0.75},
    "next_alpha_0.5":  {"target": "next", "alpha": 0.5},
    "next_alpha_0.3":  {"target": "next", "alpha": 0.3},
    "next_alpha_1.0":  {"target": "next", "alpha": 1.0},
}

init = torch.load(SBM_DIR / CFG["init_from"], map_location="cpu", weights_only=False)
BASE = init["cfg"]
OUT = SBM_DIR / CFG["out_dir"]
OUT.mkdir(parents=True, exist_ok=True)
if CFG["from_scratch"]:
    CFG["lr"], CFG["warmup_steps"] = CFG["scratch_lr"], CFG["scratch_warmup_steps"]
    print(f"FROM SCRATCH: random init (seed {CFG['seed']}), lr {CFG['lr']}, warmup {CFG['warmup_steps']}; "
          f"config from {CFG['init_from']}")
else:
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

def build_model(self_cond=False):
    if CFG["from_scratch"]:
        torch.manual_seed(CFG["seed"])         # the same random initialisation for every arm
    if BASE["backbone"] == "transformer":
        m = SlotGraphTransformer(N_MAX, BASE["n_layers"], BASE["hidden_mlp_dims"], BASE["hidden_dims"],
                                 BASE["features"], BASE["rrwp_steps"], amp=CFG["amp"], self_cond=self_cond)
        if not CFG["from_scratch"]:
            m.load_base_weights(INIT_WEIGHTS)  # a new self-conditioning channel starts at zero
    else:
        assert not self_cond, "self-conditioning needs the transformer backbone"
        m = GCNDiscrete(N_MAX)
        if not CFG["from_scratch"]:
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
- **`test_table(model)`:** the DeFoG test protocol at every step count in `eval_steps` (4, 10, 20), with the base run's sampler settings.
"""))

cells.append(code(r"""X1v, Nv = graphs_to_slots(val_g, layout, BASE["ordering"])
X1v, Nv = X1v.repeat(4, 1), Nv.repeat(4)
MASKv = layout.mask(Nv); P0v = p_slot.unsqueeze(0) * MASKv; Kv = X1v.sum(1).long()
PAIRS = [(0.3, 0.58), (0.58, 0.88), (0.88, 0.98)]

@torch.no_grad()
def _probs(model, x, t, prev=None):
    out = []
    for i in range(0, len(x), 32):
        xb, tb = x[i:i + 32], torch.full((len(x[i:i + 32]), 1), t, device=DEVICE)
        out.append(torch.sigmoid(model(xb, tb, Nv[i:i + 32]) if prev is None
                                 else model(xb, tb, Nv[i:i + 32], prev=prev[i:i + 32])))
    return torch.cat(out) * MASKv

def _guess(model, x, t, prev=None):
    return top_k_slots(_probs(model, x, t, prev), Kv, MASKv)

@torch.no_grad()
def self_correction(model, self_cond=False, seed=0):
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
        P_s = _probs(model, sample_path(torch.rand_like(X1v), P0v, X1v, tt(s)), s)
        g = top_k_slots(P_s, Kv, MASKv)
        xB = sample_path(torch.rand_like(X1v), P0v, g, tt(t))
        pB = _guess(model, xB, t, prev=P_s if self_cond else None)   # with self-conditioning the sampler passes P_s
        rows[f"{s}->{t}"] = {"random_kept": kept(xA, pA), "own_kept": kept(xB, pB),
                             "guess_right": prec(g), "output_right": prec(pB), "gain": prec(pB) - prec(g)}
    return rows

REF_TEST = mmd_metrics(test_g, train_g)

@torch.no_grad()
def test_table(model, self_cond=False):
    model.eval()
    table = {}
    for steps in CFG["eval_steps"]:
        runs = []
        for r in range(CFG["final_runs"]):
            torch.manual_seed(1000 + r)
            idx = torch.randint(len(n_train), (CFG["num_samples"],), device=DEVICE)
            n, m = n_train[idx], m_train[idx]
            x = generate(model, p_slot, n, layout, steps, final=BASE["final"], n_edges=m,
                         renoise=BASE.get("renoise", "hard"), schedule=BASE.get("schedule", "poly3"),
                         same_u=CFG.get("same_u", False),
                         self_cond=self_cond)
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

Same seed, same steps, same learning-rate schedule; the arms differ only in α. Progress is saved every 500 steps and on interrupt (■); rerunning continues.
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

def finetune(arm, ov):
    arm_dir = OUT / arm; arm_dir.mkdir(exist_ok=True)
    done_path, last_path = arm_dir / "done.pt", arm_dir / "last.pt"
    student = build_model(ov.get("self_cond", False)); ema_tgt = frozen_copy(student); ema_smp = frozen_copy(student)
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
                              coupling=BASE["coupling"], lam=BASE["lam"],
                              target=ov.get("target", BASE["target"]),
                              alpha=ov.get("alpha", BASE.get("alpha")),
                              self_cond=ov.get("self_cond", False), sc_prob=CFG["sc_prob"],
                              delta_range=tuple(ov.get("delta_range", CFG["delta_range"])))
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
    atomic_save({"cfg": {**BASE, **{k: v for k, v in ov.items() if k in ("target", "alpha", "self_cond")},
                         **{"finetune": CFG, "arm": arm, "overrides": ov}}, "n_max": N_MAX, "step": init["step"] + step,
                 "best_step": None, "val": None,
                 "weights": {k: (v.half() if v.is_floating_point() else v) for k, v in ema_smp.state_dict().items()}},
                arm_dir / f"export_{arm}.pt")
    return ema_smp

MODELS = {arm: finetune(arm, ov) for arm, ov in ARMS.items()}
"""))

cells.append(md(r"""## G. Compare"""))

cells.append(code(r"""for arm, model in MODELS.items():
    if arm not in RESULTS:
        sc = ARMS[arm].get("self_cond", False)
        RESULTS[arm] = {"self_correction": self_correction(model, sc), "test": test_table(model, sc)}
        save_results()

for name in ["start", *ARMS]:
    show_sc(RESULTS[name]["self_correction"], f"\n{name.upper()}: self-correction check")
    show_test({int(k): v for k, v in RESULTS[name]["test"].items()}, f"{name.upper()}: test set")

def g(name, steps, key):
    r = RESULTS[name]["test"]; r = r.get(steps, r.get(str(steps)))
    return r[key]
print("\nSUMMARY (lower ratio is better; higher V.U.N. is better; lower 'own kept' is better)")
head = "".join(f"{f'{st}-step ratio':>18}{f'{st}-step VUN':>13}" for st in CFG["eval_steps"])
print(f"{'':>18}{head}{'own kept .58->.88':>19}{'own kept .88->.98':>19}")
for name in ["start", *ARMS]:
    sc = RESULTS[name]["self_correction"]
    cols = "".join(f"{g(name, st, 'average_ratio')[0]:11.1f}+/-{g(name, st, 'average_ratio')[1]:<4.1f}"
                   f"{g(name, st, 'vun')[0]:13.1%}" for st in CFG["eval_steps"])
    print(f"{name:>18}{cols}{sc['0.58->0.88']['own_kept']:19.1%}{sc['0.88->0.98']['own_kept']:19.1%}")
"""))

cells.append(md(r"""## H. How to read it, and what to push

Compare every arm with **`next_alpha_0.75`** (today's setting; same training and starting point, only α differs). For the from-scratch run, `start` is the random initialisation, so its scores are meaningless. With 200 test graphs per step count, a V.U.N. gap of about 8 points or more is unlikely to be chance.

- **A lower α helps** if `next_alpha_0.5` or `next_alpha_0.3` has **higher V.U.N. at 10 and 20 steps** than `next_alpha_0.75` without worse MMDs. Then the next from-scratch run should use that α.
- **`next_alpha_1.0` (consistency off) is the key result for the paper:**
  - clearly **worse** than `next_alpha_0.75`: the consistency term helps;
  - about the **same**: the gain from `target="next"` doesn't come from consistency training, and the method is effectively a denoiser with a good sampler. That changes how the paper should be framed.
- **Collapse check for low α:** `unique` well below 1.0 means the generated graphs started to look alike.
- **4 steps:** all arms will probably stay near 0% valid; look at the 4-step ratio and MMDs for movement.

**Push the results** (small files) from the repository root:

```bash
git add Codes/SBM/results_spectre/ab_alpha_scratch/ab_results.json Codes/SBM/results_spectre/ab_alpha_scratch/*/export_*.pt
```

```bash
git commit -m "Anchor share (alpha) A/B results"
```

```bash
git push origin main
```
"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).with_name("sampler_aware_ab.ipynb")
nbformat.write(nb, out)
print("wrote", out)
