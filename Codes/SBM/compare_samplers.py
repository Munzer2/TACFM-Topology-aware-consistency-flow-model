"""
compare_samplers.py — same trained weights, different samplers, scored on the SPECTRE test set.

Samplers (see tacfm_d.py):
  ours    predict -> top-k graph -> re-noise -> predict (renoise='hard', schedule='poly3'),
          with fresh noise each step (same_u=False) or the same u throughout (same_u=True)
  defog   DeFoG's CTMC sampler: keep the current graph, flip edges toward a sampled prediction
          with rate 1/(1-t) (+ eta stochasticity, + omega target guidance), time distortion

Each config is scored with V.U.N. (fast); --full adds the five MMDs (slower, needs ORCA).

Run from Codes/SBM:
    python compare_samplers.py results_spectre/ab_self_cond/target_next/export_target_next.pt --steps 10
    python compare_samplers.py <export> --steps 4 10 20 --num 80 --configs ours ours_same_u defog_planar
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
from spectre_eval import mmd_metrics, ratios, vun_metrics
from tacfm_d import (GCNDiscrete, SlotLayout, edge_prior, generate, generate_defog, graphs_to_slots,
                     slots_to_graphs)

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent

CONFIGS = {
    "ours":            dict(kind="ours", same_u=False),
    "ours_same_u":     dict(kind="ours", same_u=True),
    "ours_uniform":    dict(kind="ours", same_u=False, schedule="uniform"),
    "ours_same_u_poly2": dict(kind="ours", same_u=True, schedule="poly2"),
    "ours_same_u_poly4": dict(kind="ours", same_u=True, schedule="poly4"),
    "ours_same_u_poly5": dict(kind="ours", same_u=True, schedule="poly5"),
    "defog_plain":     dict(kind="defog", eta=0.0, omega=0.0, distortion="identity", final="sample"),
    "defog_polydec":   dict(kind="defog", eta=0.0, omega=0.0, distortion="polydec", final="sample"),
    "defog_planar":    dict(kind="defog", eta=50.0, omega=0.05, distortion="polydec", final="sample"),
    "defog_planar_topk": dict(kind="defog", eta=50.0, omega=0.05, distortion="polydec", final="topk"),
    "defog_tg":        dict(kind="defog", eta=0.0, omega=0.5, distortion="polydec", final="sample"),
    "defog_tg_eta":    dict(kind="defog", eta=10.0, omega=0.5, distortion="polydec", final="sample"),
}


def load_model(path, device):
    ex = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ex["cfg"]
    if cfg["backbone"] == "transformer":
        m = SlotGraphTransformer(ex["n_max"], cfg["n_layers"], cfg["hidden_mlp_dims"], cfg["hidden_dims"],
                                 cfg["features"], cfg["rrwp_steps"], amp=False, self_cond=cfg.get("self_cond", False))
    else:
        m = GCNDiscrete(ex["n_max"])
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ex["weights"].items()})
    return m.to(device).eval(), ex, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("export")
    p.add_argument("--steps", type=int, nargs="+", default=[10])
    p.add_argument("--configs", nargs="+", default=list(CONFIGS))
    p.add_argument("--num", type=int, default=80, help="generated graphs per config and step count")
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--full", action="store_true", help="also compute the five MMDs and the average ratio")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    model, ex, cfg = load_model(args.export, device)
    load = lambda part: pickle.load(open(HERE / "data" / "spectre" / f"{cfg['dataset']}_{part}.pkl", "rb"))
    train_g, test_g = load("train"), load("test")
    layout = SlotLayout(ex["n_max"], device)
    x1, n_train = graphs_to_slots(train_g, layout, cfg["ordering"])
    m_train = x1.sum(1).long()
    p_slot = edge_prior(x1, layout.mask(n_train), cfg["prior"])
    ref = mmd_metrics(test_g, train_g) if args.full else None
    print(f"{args.export} | trained {ex['step']:,} steps | target {cfg.get('target')} | {args.num} graphs per cell", flush=True)

    results = {}
    for steps in args.steps:
        for name in args.configs:
            c = CONFIGS[name]
            torch.manual_seed(1000 + steps)
            gen, t0 = [], time.time()
            for b in range(0, args.num, args.batch):
                k = min(args.batch, args.num - b)
                idx = torch.randint(len(n_train), (k,), device=device)
                n, m = n_train[idx], m_train[idx]
                if c["kind"] == "ours":
                    x = generate(model, p_slot, n, layout, steps, final=cfg.get("final", "topk"), n_edges=m,
                                 renoise="hard", schedule=c.get("schedule", cfg.get("schedule", "poly3")),
                                 same_u=c["same_u"])
                else:
                    x = generate_defog(model, p_slot, n, layout, steps, eta=c["eta"], omega=c["omega"],
                                       distortion=c["distortion"], final=c["final"], n_edges=m)
                gen += slots_to_graphs(x, n, layout)
            out = vun_metrics(gen, train_g, cfg["dataset"], cfg["dataset"] == "planar")
            out["edges"] = float(np.mean([g.number_of_edges() for g in gen]))
            if args.full:
                mm = mmd_metrics(test_g, gen)
                out.update(mm)
                out.update(ratios(mm, ref))
            results[f"{name}@{steps}"] = out
            extra = (f"  deg {out['degree']:.4f} clus {out['clustering']:.3f} orb {out['orbit']:.4f} "
                     f"ratio {out['average_ratio']:.1f}") if args.full else ""
            print(f"{steps:3d} steps  {name:18s} V.U.N. {out.get('vun', float('nan')):6.1%}  valid {out.get('valid', float('nan')):6.1%}"
                  f"  unique {out['unique']:5.2f}  edges {out['edges']:6.1f}{extra}  ({time.time() - t0:.0f}s)", flush=True)

    out_path = Path(args.out) if args.out else Path(args.export).with_name(f"samplers_{Path(args.export).stem}.json")
    json.dump({"export": args.export, "num": args.num, "configs": {k: CONFIGS[k] for k in args.configs},
               "results": results}, open(out_path, "w"), indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
