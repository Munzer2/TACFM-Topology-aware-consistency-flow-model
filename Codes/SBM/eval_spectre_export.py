"""
eval_spectre_export.py — score an exported TACFM-D model (export_best.pt / export_last.pt from
notebooks/spectre_gpu_training.ipynb) on the SPECTRE test set, on CPU or GPU.

Same protocol as the notebook's Part H (DeFoG's): for each step count, --runs runs of --num_samples
generated graphs, scored against the 40 test graphs with the DiGress/DeFoG metrics; mean +/- std.
The data, node ordering and edge prior are rebuilt from the config stored in the export, so the
result matches what the notebook would report.

Run from Codes/SBM:
    python eval_spectre_export.py results_spectre/<run_name>/export_best.pt
    python eval_spectre_export.py results_spectre/<run_name>/export_last.pt --steps 4 --runs 1   # quick look
"""

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from defog_nn import SlotGraphTransformer
from spectre_eval import MMD_KEYS, mmd_metrics, ratios, vun_metrics
from tacfm_d import GCNDiscrete, SlotLayout, edge_prior, generate, graphs_to_slots, slots_to_graphs

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("export", help="export_best.pt or export_last.pt")
    p.add_argument("--steps", type=int, nargs="+", default=None, help="default: the run's eval_steps")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--num_samples", type=int, default=40)
    p.add_argument("--final", default=None, help="topk / threshold / sample (default: the run's setting)")
    p.add_argument("--renoise", default=None, help="hard / soft (default: the run's setting, else soft)")
    p.add_argument("--schedule", default=None, help="uniform / polyP (default: the run's setting, else uniform)")
    p.add_argument("--no_orbit", action="store_true", help="skip the orbit MMD (needs the ORCA binary)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ex = torch.load(args.export, map_location="cpu", weights_only=False)
    cfg = ex["cfg"]
    steps_list = args.steps or cfg["eval_steps"]
    final = args.final or cfg["final"]
    renoise = args.renoise or cfg.get("renoise", "soft")
    schedule = args.schedule or cfg.get("schedule", "uniform")
    keys = tuple(k for k in MMD_KEYS if not (k == "orbit" and (args.no_orbit or not cfg.get("orbit", True))))
    print(f"{cfg['run_name']} | trained {ex['step']:,} steps (best so far at {ex['best_step']}) | "
          f"final='{final}' renoise='{renoise}' schedule='{schedule}' | {args.runs} runs x {args.num_samples} samples | device {device}")

    load = lambda part: pickle.load(open(HERE / "data" / "spectre" / f"{cfg['dataset']}_{part}.pkl", "rb"))
    train_g, test_g = load("train"), load("test")
    layout = SlotLayout(ex["n_max"], device)
    x1_train, n_train = graphs_to_slots(train_g, layout, cfg["ordering"])
    m_train = x1_train.sum(1).long()
    p_slot = edge_prior(x1_train, layout.mask(n_train), cfg["prior"])

    if cfg["backbone"] == "transformer":
        model = SlotGraphTransformer(ex["n_max"], cfg["n_layers"], cfg["hidden_mlp_dims"], cfg["hidden_dims"],
                                     cfg["features"], cfg["rrwp_steps"], amp=False)
    else:
        model = GCNDiscrete(ex["n_max"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ex["weights"].items()})
    model.to(device).eval()

    ref = mmd_metrics(test_g, train_g, keys)
    check_validity = cfg["dataset"] == "planar"
    if not check_validity:
        try:
            import graph_tool  # noqa: F401
            check_validity = True
        except ImportError:
            print("graph_tool not installed: V.U.N. reported without the SBM validity check")

    table = {}
    for steps in steps_list:
        rows, t0 = [], time.time()
        for r in range(args.runs):
            torch.manual_seed(1000 + r)
            idx = torch.randint(len(n_train), (args.num_samples,), device=device)
            n, m = n_train[idx], m_train[idx]
            x = generate(model, p_slot, n, layout, steps, final=final, n_edges=m, renoise=renoise, schedule=schedule)
            gen = slots_to_graphs(x, n, layout)
            out = mmd_metrics(test_g, gen, keys)
            out.update(ratios(out, ref))
            out.update(vun_metrics(gen, train_g, cfg["dataset"], check_validity))
            rows.append(out)
        table[steps] = {k: (float(np.mean([x[k] for x in rows])), float(np.std([x[k] for x in rows])))
                        for k in rows[0]}
        print(f"  {steps:2d}-step ({time.time() - t0:.0f}s)  " + "  ".join(
            f"{k}={v[0]:.4f}+/-{v[1]:.4f}" for k, v in table[steps].items()
            if k in ("degree", "clustering", "orbit", "spectre", "wavelet", "average_ratio", "valid", "vun")),
            flush=True)

    out_path = Path(args.export).with_name(f"cpu_eval_{Path(args.export).stem}_step{ex['step']}_{renoise}_{schedule}.json")
    json.dump({"export": str(args.export), "step": ex["step"], "final": final, "renoise": renoise,
               "schedule": schedule, "runs": args.runs,
               "num_samples": args.num_samples, "reference": ref,
               "test": {str(s): v for s, v in table.items()}}, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
