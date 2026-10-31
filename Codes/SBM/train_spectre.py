"""
train_spectre.py — TACFM-D on the SPECTRE benchmarks (Planar, SBM).

Data:        data/spectre/{planar,sbm}_{train,val,test}.pkl  (DiGress/DeFoG split, 128/32/40;
             build with  python -I data/prepare_spectre.py)
Model:       tacfm_d.py (discrete path, GCN, consistency training, degree-sequence coupling)
Evaluation:  spectre_eval (DiGress/DeFoG code): degree / clustering / orbit / spectre / wavelet
             MMD, their ratios to the train-vs-reference MMD, and V.U.N.
             - during training: 40 samples vs the 32 validation graphs, every --eval_every epochs
             - at the end: the checkpoint with the best validation average ratio, scored on the
               40 test graphs, --final_runs runs of 40 samples each (mean +/- std), as in DeFoG

Run from Codes/SBM:
    python train_spectre.py --dataset planar --epochs 50 --eval_every 25     # smoke test
    python train_spectre.py --dataset planar --epochs 2000                   # real run
"""

import argparse
import copy
import json
import os
import pickle
import time
import warnings

import numpy as np
import torch

from tacfm_d import (GCNDiscrete, SlotLayout, batch_loss, edge_prior, generate,
                     graphs_to_slots, slots_to_graphs)
from spectre_eval import mmd_metrics, ratios, vun_metrics

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(name, part):
    return pickle.load(open(os.path.join("data", "spectre", f"{name}_{part}.pkl"), "rb"))


def score(gen_graphs, ref_graphs, ref_mmd, train_graphs, dataset, check_validity):
    out = mmd_metrics(ref_graphs, gen_graphs)
    out.update(ratios(out, ref_mmd))
    out.update(vun_metrics(gen_graphs, train_graphs, dataset, check_validity))
    return out


def sample_graphs(model, p_slot, n_train, layout, num, steps):
    n_nodes = n_train[torch.randint(len(n_train), (num,), device=n_train.device)]
    x = generate(model, p_slot, n_nodes, layout, steps)
    return slots_to_graphs(x, n_nodes, layout)


def fmt(r):
    keys = [("degree", "deg"), ("clustering", "clus"), ("orbit", "orb"), ("spectre", "spec"),
            ("wavelet", "wav"), ("average_ratio", "ratio"), ("valid", "valid"), ("vun", "VUN")]
    return "  ".join(f"{short}={r[k]:.4f}" for k, short in keys if k in r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["planar", "sbm"], default="planar")
    p.add_argument("--ordering", choices=["bfs", "degree", "none"], default="bfs")
    p.add_argument("--prior", choices=["slot", "global"], default="slot")
    p.add_argument("--coupling", choices=["structure", "independent"], default="structure")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--ema", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=3.0)
    p.add_argument("--t_dist", choices=["uniform", "polydec", "late_mix"], default="uniform",
                   help="training-time distribution of t (see tacfm_d.train_times)")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, nargs="+", default=[1, 4, 10])
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--num_samples", type=int, default=40)
    p.add_argument("--final_runs", type=int, default=5)
    p.add_argument("--no_validity", action="store_true",
                   help="skip the validity check (SBM validity needs graph_tool, Linux/macOS only)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    tag = f"{args.dataset}_{args.ordering}_{args.prior}_{args.coupling}_s{args.seed}"
    if args.t_dist != "uniform":
        tag += f"_t{args.t_dist}"
    out_dir = args.out or os.path.join("results_spectre", tag)
    os.makedirs(out_dir, exist_ok=True)
    check_validity = not args.no_validity

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_g, val_g, test_g = (load(args.dataset, s) for s in ("train", "val", "test"))
    n_max = max(G.number_of_nodes() for G in train_g + val_g + test_g)
    layout = SlotLayout(n_max, DEVICE)
    x1_train, n_train = graphs_to_slots(train_g, layout, args.ordering)
    mask_train = layout.mask(n_train)
    p_slot = edge_prior(x1_train, mask_train, args.prior)

    print(f"{tag} | device={DEVICE} | train/val/test {len(train_g)}/{len(val_g)}/{len(test_g)} | "
          f"N_MAX={n_max} D={layout.d} | edge prior mean {float((p_slot * mask_train).sum() / mask_train.sum()):.4f}",
          flush=True)

    ref = {"val": mmd_metrics(val_g, train_g), "test": mmd_metrics(test_g, train_g)}
    print("reference MMD (train vs val):", {k: round(v, 5) for k, v in ref["val"].items()}, flush=True)

    student = GCNDiscrete(n_max, node_hidden=args.hidden, num_gcn_layers=args.layers).to(DEVICE)
    ema = copy.deepcopy(student).eval()
    for q in ema.parameters():
        q.requires_grad = False
    opt = torch.optim.Adam(student.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"parameters: {sum(q.numel() for q in student.parameters()):,}", flush=True)

    history, best = [], {"ratio": float("inf"), "epoch": None}
    coupling = args.coupling == "structure"
    start = time.time()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(len(x1_train), device=DEVICE)
        running, nb = 0.0, 0
        for k in range(0, len(perm), args.batch_size):
            b = perm[k:k + args.batch_size]
            loss = batch_loss(student, ema, x1_train[b], n_train[b], p_slot, layout, coupling, args.lam,
                              t_dist=args.t_dist)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            opt.step()
            with torch.no_grad():
                for pe, ps in zip(ema.parameters(), student.parameters()):
                    pe.mul_(args.ema).add_(ps, alpha=1 - args.ema)
            running += loss.item()
            nb += 1
        sched.step()
        entry = {"epoch": ep, "loss": running / nb, "secs": time.time() - start}

        if ep % args.eval_every == 0 or ep == args.epochs:
            rng = torch.get_rng_state()           # evaluation must not change the training stream
            entry["val"] = {}
            for steps in args.steps:
                gen = sample_graphs(student, p_slot, n_train, layout, args.num_samples, steps)
                entry["val"][steps] = score(gen, val_g, ref["val"], train_g, args.dataset, check_validity)
                print(f"  ep {ep:5d} {steps:2d}-step  {fmt(entry['val'][steps])}", flush=True)
            torch.set_rng_state(rng)
            sel = entry["val"][max(args.steps)]["average_ratio"]
            if sel < best["ratio"]:
                best = {"ratio": sel, "epoch": ep}
                torch.save(student.state_dict(), os.path.join(out_dir, "best_model.pt"))
        history.append(entry)
        if ep % 10 == 0 or "val" in entry:
            print(f"ep {ep:5d}  loss {entry['loss']:.4f}  ({entry['secs']:.0f}s)", flush=True)

    torch.save(student.state_dict(), os.path.join(out_dir, "final_model.pt"))

    # ---- final test evaluation, best-on-validation checkpoint ----
    student.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt"), weights_only=True))
    print(f"\nTEST (checkpoint from epoch {best['epoch']}, {args.final_runs} runs x {args.num_samples} samples)")
    test = {}
    for steps in args.steps:
        runs = []
        for r in range(args.final_runs):
            torch.manual_seed(1000 + r)
            gen = sample_graphs(student, p_slot, n_train, layout, args.num_samples, steps)
            runs.append(score(gen, test_g, ref["test"], train_g, args.dataset, check_validity))
            if r == 0:
                adjs = [np.asarray(nx_to_dense(G)) for G in gen]
                np.savez_compressed(os.path.join(out_dir, f"samples_{steps}step.npz"), *adjs)
        test[steps] = {k: {"mean": float(np.mean([x[k] for x in runs])), "std": float(np.std([x[k] for x in runs]))}
                       for k in runs[0]}
        print(f"  {steps:2d}-step  " + "  ".join(f"{k}={v['mean']:.4f}+/-{v['std']:.4f}"
                                                 for k, v in test[steps].items()
                                                 if k in ("degree", "clustering", "orbit", "spectre", "wavelet",
                                                          "average_ratio", "valid", "vun")), flush=True)

    json.dump({"config": vars(args), "n_max": n_max, "reference": ref, "best_epoch": best["epoch"],
               "history": history, "test": test}, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    print(f"\nwrote {out_dir}/results.json")


def nx_to_dense(G):
    import networkx as nx
    return nx.to_numpy_array(G, dtype=np.int8)


if __name__ == "__main__":
    main()
