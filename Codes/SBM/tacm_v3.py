"""
tacm_v3.py — consistency training for graphs with the sphere removed.

WHY THE SPHERE GOES
Every real graph normalises to a point with all 190 entries at exactly +/-1/sqrt(190).
So graphs occupy 2^190 isolated corners of a hypercube inscribed in S^189, while the
sphere itself is a continuum. Graphs are a MEASURE-ZERO subset: a random point on the
sphere is, with probability 1, not a graph. "Stay on the sphere" and "stay a graph" are
therefore unrelated constraints -- unlike RCM's SO(3), where every point of the manifold
IS a valid rotation. The sphere constrains nothing that matters, so it goes.

WHAT REPLACES IT: a probability path, sampled to binary.

    u   = rand(B, 190)                      # ONE draw, fixed per sample
    p_t = (1-t) * p_marg + t * A_true       # interpolate PROBABILITIES
    x_t = (u < p_t)                         # a REAL BINARY GRAPH at every t

  t=0   -> p_0 = p_marg          -> x_0 is a random graph at the data's own density
  t=0.5 -> half corrupted        -> still a real binary graph
  t=1   -> p_1 = A_true in {0,1} -> x_1 == A_true EXACTLY (boundary condition free)

The shared u makes the trajectory deterministic given (u, A_true), so "two points on the
same path" still means something and consistency training still applies with no teacher.
p_marg is per-slot, computed from training data, so it encodes the padding structure and
x_0 automatically has a realistic node-count and density profile.

FIXES vs _TACM_.py
1. Representation. Intermediates are real binary graphs, never float mush. Topology is
   never dissolved, so it never has to be reconstructed.
2. Loss. arccos^2 is a regression loss grading a binary decision, and it is provably
   miscalibrated: with x1 = +0.0725, predicting +0.20 (CORRECT edge) costs 0.01626 while
   predicting -0.01 (WRONG) costs 0.00681 -- the right answer is penalised 2.4x more than
   the wrong one. A perfect graph with different magnitudes scores 0.18, not 0. Replaced
   with BCE on logits, which grades the decision actually being made.
3. GCN message passing. sigmoid(adj*10) on sphere-shrunk entries gave a 2.07x edge/
   non-edge contrast with only 32.6% of aggregation weight on real neighbours. Now the
   adjacency IS binary, so message passing runs on the true graph. The bug does not get
   fixed; it ceases to exist.
4. Time range. Training sampled t in [0.3, 0.95] but generation evaluates at t=0 -- never
   trained. Now t in [0, 1].

DELIBERATELY UNCHANGED (to isolate the representation as the variable)
  - node features are still the raw adjacency row, as in model_GCN.py. Not permutation-
    equivariant, but Yan (2025) argues non-invariant generators can beat invariant ones,
    so this is contested rather than clearly wrong. Keeping it constant across arms.
  - no structural loss term. DeFoG and BWFlow add none: fix the representation and let
    structure be learned. It also avoids optimising degree/triangles and then reporting
    degree/clustering MMD, which is close to circular.

STILL UNSOLVED
  The path is linear in probability space, so each edge still moves on its own scalar
  schedule. BWFlow's flat-then-cliff critique may well still apply. Fixing that needs
  their Bures-Wasserstein construction.

Baseline arm is _TACM_.py verbatim. Both arms: clean data, corrected metrics, held-out
evaluation.
"""

import argparse
import copy
import json
import os
import pickle
import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model_GCN import SinusoidalTimeEmbeddings, flatten_adj_to_vec, vect_to_adj, normalize_to_sphere
from _TACM_ import _TACM_, geodesic_interpolation
from evaluate import adj_to_graphs
from metrics_v2 import degree_stats, clustering_stats, spectral_stats

MAX_NODES = 20
DATA_DIM = MAX_NODES * (MAX_NODES - 1) // 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
#  BACKBONE — identical to model_GCN.GCN_TACFM except the adjacency is already binary
# ============================================================

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, h, adj):
        N = h.shape[1]
        adj_hat = adj + torch.eye(N, device=adj.device).unsqueeze(0)
        deg = adj_hat.sum(-1, keepdim=True).clamp(min=1.0)
        h_out = self.linear(torch.bmm(adj_hat / deg, h)) + self.skip(h)
        return self.act(self.norm(h_out))


class GCN_Discrete(nn.Module):
    """Takes a BINARY adjacency. Returns per-edge logits."""

    def __init__(self, max_nodes=20, time_dim=32, node_hidden=128, num_gcn_layers=4):
        super().__init__()
        self.max_nodes = max_nodes
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(time_dim), nn.Linear(time_dim, time_dim), nn.GELU())
        self.node_input = nn.Linear(max_nodes + time_dim, node_hidden)
        self.gcn_layers = nn.ModuleList(
            [GCNLayer(node_hidden, node_hidden) for _ in range(num_gcn_layers)])
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_hidden, 2 * node_hidden), nn.GELU(),
            nn.Linear(2 * node_hidden, node_hidden), nn.GELU(),
            nn.Linear(node_hidden, 1))

    def forward(self, x_flat, t):
        N = self.max_nodes
        adj = vect_to_adj(x_flat, n=N)          # already binary — no sigmoid(adj*10)
        node_feat = adj                          # raw row, same as the original
        t_emb = self.time_mlp(t).unsqueeze(1).expand(-1, N, -1)
        h = self.node_input(torch.cat([node_feat, t_emb], dim=-1))
        for gcn in self.gcn_layers:
            h = gcn(h, adj)                      # message passing on the TRUE graph
        idx = torch.triu_indices(N, N, offset=1)
        edge_feat = torch.cat([h[:, idx[0], :], h[:, idx[1], :]], dim=-1)
        return self.edge_mlp(edge_feat).squeeze(-1)   # logits (B, 190)


# ============================================================
#  DISCRETE PROBABILITY PATH
# ============================================================

def sample_path(u, p_marg, x1_bin, t):
    """x_t = 1[u < (1-t)*p_marg + t*x1]. Binary graph. Exact at t=0 and t=1."""
    p_t = (1 - t) * p_marg.unsqueeze(0) + t * x1_bin
    return (u < p_t).float()


# ============================================================
#  TRAINING
# ============================================================

def load_tensors(path, max_nodes=MAX_NODES):
    graphs = pickle.load(open(path, "rb"))
    adj = []
    for G in graphs:
        a = nx.to_numpy_array(G)
        p = np.zeros((max_nodes, max_nodes))
        p[:a.shape[0], :a.shape[0]] = a
        adj.append(p)
    return torch.tensor(np.array(adj, dtype=np.float32)), graphs


def train(mode, loader, p_marg, epochs, lr, seed, ema_rate=0.995):
    torch.manual_seed(seed)
    np.random.seed(seed)

    student = (_TACM_(max_nodes=MAX_NODES) if mode == "sphere" else GCN_Discrete(MAX_NODES)).to(DEVICE)
    ema = copy.deepcopy(student).eval()
    for p in ema.parameters():
        p.requires_grad = False

    opt = optim.Adam(student.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    start = time.time()

    for _ in range(epochs):
        for (adj_batch,) in loader:
            adj_batch = adj_batch.to(DEVICE)
            B = adj_batch.shape[0]
            x1_bin = flatten_adj_to_vec(adj_batch)              # (B, 190) in {0,1}

            delta = torch.rand(B, 1, device=DEVICE) * 0.2 + 0.1

            if mode == "sphere":
                x1 = normalize_to_sphere(x1_bin * 2 - 1)
                x0 = normalize_to_sphere(torch.randn_like(x1))
                t = torch.rand(B, 1, device=DEVICE) * 0.65 + 0.3    # original range
                t_minus = (t - delta).clamp(min=0.01)
                x_t = geodesic_interpolation(x0, x1, t)
                x_t_minus = geodesic_interpolation(x0, x1, t_minus)

                pred_t = student(x_t, t)
                with torch.no_grad():
                    pred_tm = ema(x_t_minus, t_minus)
                cos_c = (pred_t * pred_tm).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)
                cos_d = (pred_t * x1).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)
                loss = torch.mean(torch.acos(cos_c) ** 2) + 3.0 * torch.mean(torch.acos(cos_d) ** 2)
            else:
                u = torch.rand(B, DATA_DIM, device=DEVICE)          # shared randomness
                t = torch.rand(B, 1, device=DEVICE)                 # fix 4: full [0,1]
                t_minus = (t - delta).clamp(min=0.0)
                x_t = sample_path(u, p_marg, x1_bin, t)             # BINARY graph
                x_t_minus = sample_path(u, p_marg, x1_bin, t_minus) # BINARY graph

                logit_t = student(x_t, t)
                with torch.no_grad():
                    soft_target = torch.sigmoid(ema(x_t_minus, t_minus))
                loss = F.binary_cross_entropy_with_logits(logit_t, soft_target) \
                     + 3.0 * F.binary_cross_entropy_with_logits(logit_t, x1_bin)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            opt.step()
            with torch.no_grad():
                for pe, ps in zip(ema.parameters(), student.parameters()):
                    pe.data.mul_(ema_rate).add_(ps.data, alpha=1 - ema_rate)
        sched.step()

    return student, time.time() - start


@torch.no_grad()
def generate(model, mode, p_marg, n, steps, max_nodes=MAX_NODES):
    model.eval()
    dim = max_nodes * (max_nodes - 1) // 2
    if mode == "sphere":
        x = normalize_to_sphere(torch.randn(n, DATA_DIM, device=DEVICE))
        if steps == 1:
            x = model(x, torch.zeros(n, 1, device=DEVICE))
        else:
            for i, tv in enumerate(torch.linspace(0.0, 0.8, steps)):
                x = normalize_to_sphere(model(x, torch.full((n, 1), tv.item(), device=DEVICE)))
                if i < steps - 1:
                    x = normalize_to_sphere(x + torch.randn_like(x) * 0.05)
        out = (vect_to_adj(x, n=MAX_NODES) > 0).float()
    else:
        x = (torch.rand(n, dim, device=DEVICE) < p_marg.unsqueeze(0)).float()
        ts = torch.linspace(0.0, 1.0, steps + 1)[:-1]
        for i, tv in enumerate(ts):
            A_hat = torch.sigmoid(model(x, torch.full((n, 1), tv.item(), device=DEVICE)))
            if i == len(ts) - 1:
                x = (A_hat > 0.5).float()
            else:
                t_next = ts[i + 1].item()
                p = (1 - t_next) * p_marg.unsqueeze(0) + t_next * A_hat
                x = (torch.rand_like(p) < p).float()
        out = vect_to_adj(x, n=max_nodes)
    model.train()
    return out.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--modes", nargs="+", default=["sphere", "discrete"])
    p.add_argument("--train_path", default="data/community_v2_train.pkl")
    p.add_argument("--test_path", default="data/community_v2_test.pkl")
    p.add_argument("--out", default="results_tacm_v3/results.json")
    args = p.parse_args()

    train_adj, _ = load_tensors(args.train_path)
    _, test_graphs = load_tensors(args.test_path)
    loader = DataLoader(TensorDataset(train_adj), batch_size=16, shuffle=True)

    # per-slot marginal: encodes padding structure, so x_0 has a realistic size profile
    p_marg = flatten_adj_to_vec(train_adj).mean(0).to(DEVICE)

    print(f"device={DEVICE} epochs={args.epochs} seed={args.seed} test={len(test_graphs)} (held out)")
    print(f"reference: nodes {np.mean([g.number_of_nodes() for g in test_graphs]):.1f}, "
          f"edges {np.mean([g.number_of_edges() for g in test_graphs]):.1f}")
    print(f"p_marg: mean {p_marg.mean():.3f}, range [{p_marg.min():.3f}, {p_marg.max():.3f}]\n")

    results = {}
    for mode in args.modes:
        model, secs = train(mode, loader, p_marg, args.epochs, args.lr, args.seed)
        results[mode] = {}
        for steps in (1, 4):
            gen = adj_to_graphs(generate(model, mode, p_marg, args.num_samples, steps))
            if not gen:
                print(f"  {mode:9s} {steps}-step  NO VALID GRAPHS")
                results[mode][f"{steps}-step"] = None
                continue
            r = {"deg": degree_stats(test_graphs, gen),
                 "clus": clustering_stats(test_graphs, gen),
                 "spec": spectral_stats(test_graphs, gen),
                 "edges": float(np.mean([g.number_of_edges() for g in gen])),
                 "valid": len(gen)}
            results[mode][f"{steps}-step"] = r
            print(f"  {mode:9s} {steps}-step  deg={r['deg']:.4f} clus={r['clus']:.4f} "
                  f"spec={r['spec']:.4f}  edges={r['edges']:.1f} valid={r['valid']} ({secs:.0f}s)")

    print("\n" + "=" * 70)
    print(f"{'Model':<20}{'Deg.':>11}{'Clus.':>11}{'Spec.':>11}{'Edges':>9}")
    print("-" * 70)
    for mode in args.modes:
        for k, r in results[mode].items():
            if r:
                print(f"{mode + ' ' + k:<20}{r['deg']:>11.4f}{r['clus']:>11.4f}"
                      f"{r['spec']:>11.4f}{r['edges']:>9.1f}")
    print("=" * 70)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"config": vars(args), "results": results}, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
