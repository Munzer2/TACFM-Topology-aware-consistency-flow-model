"""
g2b_model_schedule.py — G2b: choose sampling step times from the MODEL's own error profile.

G2 (g2_tc_schedule.py) showed that the data's factorization error is not what limits our sampler:
TC-optimal (front-loaded) schedules gave 0% V.U.N. vs poly3's 27.5% at 10 steps. G2b measures where
the network itself makes mistakes and where a sampling step repairs them. Everything uses the
same-u, hard top-k sampler of tacfm_d.generate and the 32 validation graphs (test set untouched).

  maps    e(t):      quality of the model's top-k guess from a state on the real noisy path at t
                     (wrong edges = true edges missing from the guess; |guess| = |true edges|),
                     whether the guess is already valid (planar + connected), and how many of the
                     already-revealed true edges it drops / how many noise edges it keeps.
          r(t->t'):  the sampler's step: guess g_t, re-noise with the SAME u to t', predict again.
                     gain = wrong(g_t) - wrong(g_t'); validity before/after.
                     (Approximation: g_t is made from the real path, not from earlier sampler steps.)
  plan    DP over the grid maximizing the summed gain of K-1 sampler steps (first guess at t=0),
          plus a parametric family t_k = T_last * (1 - (1 - k/(K-1))^p), scored by the validity of
          80 generated graphs on SEARCH seeds (2000+), never the test seeds (1000+).
  eval    winners vs poly3 on the test set, 5 x 40, same seeds as every other evaluation.

Run from Codes/SBM:
    python g2b_model_schedule.py maps   --export <export.pt>
    python g2b_model_schedule.py search --export <export.pt> --steps 10 20
    python g2b_model_schedule.py eval   --export <export.pt> --steps 10 20
"""

import argparse
import json
import pickle
import time
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from defog_nn import SlotGraphTransformer
from tacfm_d import SlotLayout, edge_prior, generate, graphs_to_slots, slots_to_graphs, step_times, top_k_slots

warnings.filterwarnings("ignore")
OUT = Path("results_spectre/g2b")


def load_model(export):
    ex = torch.load(export, map_location="cpu", weights_only=False)
    cfg = ex["cfg"]
    m = SlotGraphTransformer(ex["n_max"], cfg["n_layers"], cfg["hidden_mlp_dims"], cfg["hidden_dims"],
                             cfg["features"], cfg["rrwp_steps"], amp=False)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ex["weights"].items()})
    return m.eval(), ex, cfg


def setup(export):
    model, ex, cfg = load_model(export)
    load = lambda part: pickle.load(open(f"data/spectre/{cfg['dataset']}_{part}.pkl", "rb"))
    layout = SlotLayout(ex["n_max"], "cpu")
    x_tr, n_tr = graphs_to_slots(load("train"), layout, cfg["ordering"])
    p_slot = edge_prior(x_tr, layout.mask(n_tr), cfg["prior"])
    x_va, n_va = graphs_to_slots(load("val"), layout, cfg["ordering"])
    return model, layout, p_slot, x_tr, n_tr, x_va, n_va, load


@torch.no_grad()
def predict(model, x, t, n, layout, m, bs=64):
    out = []
    for i in range(0, len(x), bs):
        tt = torch.full((len(x[i:i + bs]), 1), float(t))
        out.append(torch.sigmoid(model(x[i:i + bs], tt, n[i:i + bs])))
    A = torch.cat(out) * layout.mask(n)
    return top_k_slots(A, m, layout.mask(n))


def valid_frac(g, n, layout):
    gs = slots_to_graphs(g, n, layout)
    return float(np.mean([nx.is_connected(G) and nx.check_planarity(G)[0] for G in gs]))


def wrong(g, x1):
    return ((x1 == 1) & (g == 0)).sum(1).float()          # missing true edges (= invented, since |g| = |x1|)


def cmd_maps(args):
    torch.manual_seed(0); torch.set_num_threads(args.threads)
    model, layout, p_slot, _, _, x_va, n_va, _ = setup(args.export)
    R = args.reps
    x1, n = x_va.repeat(R, 1), n_va.repeat(R)
    m = x1.sum(1).long()
    p0 = p_slot.unsqueeze(0) * layout.mask(n)
    u = torch.rand_like(x1)
    path = lambda t: (u < (1 - t) * p0 + t * x1).float()
    grid = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.999]
    res = {"grid": grid, "e": {}, "r": {}}
    guesses, t0 = {}, time.time()
    for t in grid:
        xt = path(t); g = predict(model, xt, t, n, layout, m)
        guesses[t] = g
        revealed = (xt == 1) & (x1 == 1)
        noise = (xt == 1) & (x1 == 0)
        res["e"][str(t)] = {"wrong": float(wrong(g, x1).mean()), "valid": valid_frac(g, n, layout),
                            "dropped_revealed": float(((g == 0) & revealed).sum(1).float().mean()),
                            "kept_noise": float(((g == 1) & noise).sum(1).float().mean()),
                            "true_edges": float(m.float().mean())}
        e = res["e"][str(t)]
        print(f"e(t={t:.3f}): wrong {e['wrong']:5.1f}/{e['true_edges']:.0f}  valid {e['valid']:.3f}  "
              f"dropped revealed {e['dropped_revealed']:4.1f}  kept noise {e['kept_noise']:4.1f}  ({time.time() - t0:.0f}s)", flush=True)
    for i, t in enumerate(grid[:-1]):
        g = guesses[t]
        w0 = wrong(g, x1)
        for t2 in grid[i + 1:]:
            xs = (u < (1 - t2) * p0 + t2 * g).float()          # sampler's re-noise, same u
            g2 = predict(model, xs, t2, n, layout, m)
            w1 = wrong(g2, x1)
            res["r"][f"{t}->{t2}"] = {"gain": float((w0 - w1).mean()), "wrong_after": float(w1.mean()),
                                      "valid_before": res["e"][str(t)]["valid"], "valid_after": valid_frac(g2, n, layout)}
        row = " ".join(f"{t2:g}:{res['r'][f'{t}->{t2}']['gain']:+.1f}" for t2 in grid[i + 1:])
        print(f"r(from {t:.3f}) gain in wrong edges -> {row}  ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "maps.json", "w"), indent=1)
    print("wrote", OUT / "maps.json")


def dp_from_gain(maps, K):
    grid = maps["grid"]; n = len(grid)
    gain = np.full((n, n), -np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            gain[i, j] = maps["r"][f"{grid[i]}->{grid[j]}"]["gain"]
    best = np.full((K, n), -np.inf); arg = np.zeros((K, n), int)
    best[0, 0] = 0.0                                        # first guess at t = 0
    for k in range(1, K):
        for j in range(1, n):
            cand = best[k - 1, :j] + gain[:j, j]
            i = int(np.argmax(cand)); best[k, j], arg[k, j] = cand[i], i
    j = int(np.argmax(best[K - 1])); times = [j]
    for k in range(K - 1, 0, -1):
        j = arg[k, j]; times.append(j)
    return [grid[i] for i in times[::-1]], float(best[K - 1].max())


def family(K, p, T_last):
    return [float(T_last * (1 - (1 - k / (K - 1)) ** p)) for k in range(K)]


def geometric(K, n_early, t_split, T_last=0.999):
    """Model-derived (from maps.json): early steps never create validity, the model's remaining
    errors shrink about geometrically in (1 - t) for t >= 0.8, and validity is repaired only there.
    So: n_early uniform steps on [0, t_split), then K - n_early steps with (1 - t) spaced
    geometrically from (1 - t_split) down to (1 - T_last)."""
    early = [t_split * k / n_early for k in range(n_early)]
    L = K - n_early
    late = [float(1 - (1 - t_split) * ((1 - T_last) / (1 - t_split)) ** (j / (L - 1))) for j in range(L)]
    return [float(v) for v in early + late]


@torch.no_grad()
def gen_valid(model, layout, p_slot, n_tr, m_tr, times, seeds, num):
    v = []
    for s in seeds:
        torch.manual_seed(s)
        idx = torch.randint(len(n_tr), (num,)); n, m = n_tr[idx], m_tr[idx]
        x = generate(model, p_slot, n, layout, len(times), final="topk", n_edges=m, times=times, renoise="hard", same_u=True)
        v.append(valid_frac(x, n, layout))
    return float(np.mean(v))


def cmd_search(args):
    torch.set_num_threads(args.threads)
    model, layout, p_slot, x_tr, n_tr, _, _, _ = setup(args.export)
    m_tr = x_tr.sum(1).long()
    maps = json.load(open(OUT / "maps.json"))
    out_path = OUT / "search.json"
    res = json.load(open(out_path)) if out_path.exists() else {}
    seeds = [2000 + r for r in range(args.runs)]
    for K in args.steps:
        # (The DP on 'gain in wrong edges vs the original graph' was dropped: that gain is negative
        #  everywhere because the sampler repairs planarity by moving to a DIFFERENT nearby valid graph.)
        cands = {"poly3": [float(v) for v in step_times(K, "poly3")]}
        for p in (3, 4, 6):
            for T in (0.99, 0.999):
                cands[f"fam_p{p}_T{T}"] = family(K, p, T)
        early_opts = (2, 3, 4) if K <= 10 else (3, 6)
        for n_early in early_opts:
            for t_split in (0.8, 0.85):
                cands[f"geo_e{n_early}_s{t_split}"] = geometric(K, n_early, t_split)
        for name, times in cands.items():
            key = f"{K}:{name}"
            if key in res:
                continue
            t0 = time.time()
            res[key] = {"times": times, "valid": gen_valid(model, layout, p_slot, n_tr, m_tr, times, seeds, args.num)}
            print(f"K={K:>2} {name:<18} valid {res[key]['valid']:.3f}  times {[round(v, 3) for v in times]}  ({time.time() - t0:.0f}s)", flush=True)
            json.dump(res, open(out_path, "w"), indent=1)
    print("wrote", out_path)


def cmd_eval(args):
    from spectre_eval import mmd_metrics, ratios, vun_metrics
    torch.set_num_threads(args.threads)
    model, layout, p_slot, x_tr, n_tr, _, _, load = setup(args.export)
    m_tr = x_tr.sum(1).long()
    train_g, test_g = load("train"), load("test")
    ref = mmd_metrics(test_g, train_g)
    search = json.load(open(OUT / "search.json"))
    out_path = OUT / "eval_test.json"
    res = json.load(open(out_path)) if out_path.exists() else {}
    for K in args.steps:
        fam = {k: v for k, v in search.items() if k.startswith(f"{K}:fam_")}
        geo = {k: v for k, v in search.items() if k.startswith(f"{K}:geo_")}
        best_fam = max(fam, key=lambda k: fam[k]["valid"])
        best_geo = max(geo, key=lambda k: geo[k]["valid"])
        picks = {"poly3": search[f"{K}:poly3"]["times"], "best_family": search[best_fam]["times"],
                 "best_geometric": search[best_geo]["times"]}
        for name, times in picks.items():
            key = f"{K}:{name}"
            if key in res:
                continue
            rows, t0 = [], time.time()
            for r in range(args.runs):
                torch.manual_seed(1000 + r)
                idx = torch.randint(len(n_tr), (args.num,)); n, m = n_tr[idx], m_tr[idx]
                x = generate(model, p_slot, n, layout, K, final="topk", n_edges=m, times=times, renoise="hard", same_u=True)
                gen = slots_to_graphs(x, n, layout)
                s = mmd_metrics(test_g, gen); s.update(ratios(s, ref)); s.update(vun_metrics(gen, train_g, "planar", True))
                rows.append(s)
            res[key] = {"times": times, "from": {"best_family": best_fam, "best_geometric": best_geo}.get(name, name),
                        **{k: [float(np.mean([x[k] for x in rows])), float(np.std([x[k] for x in rows]))] for k in rows[0]}}
            q = res[key]
            print(f"K={K:>2} {name:<12} V.U.N. {q['vun'][0]:.3f}±{q['vun'][1]:.3f}  deg {q['degree'][0]:.4f} clus {q['clustering'][0]:.4f} "
                  f"orb {q['orbit'][0]:.4f} spec {q['spectre'][0]:.4f} wav {q['wavelet'][0]:.4f}  times {[round(v, 3) for v in times]}  ({time.time() - t0:.0f}s)", flush=True)
            json.dump(res, open(out_path, "w"), indent=1)
    print("wrote", out_path)


def main():
    global OUT
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("maps", "search", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--export", required=True)
        s.add_argument("--threads", type=int, default=16)
        s.add_argument("--steps", type=int, nargs="+", default=[10, 20])
        s.add_argument("--reps", type=int, default=4)
        s.add_argument("--runs", type=int, default=2 if name == "search" else 5)
        s.add_argument("--num", type=int, default=40)
        s.add_argument("--out", default=str(OUT), help="output folder (default results_spectre/g2b)")
    args = p.parse_args()
    OUT = Path(args.out)
    {"maps": cmd_maps, "search": cmd_search, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
