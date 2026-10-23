"""
tacfm_d.py — TACFM-D (discrete consistency model) for graphs of any size.

Same method as tacm_v3.py's discrete arm and coupling_experiment.py's 'structure' arm, generalised:

  * any N_MAX (graphs zero-padded to N_MAX nodes; D = N_MAX(N_MAX-1)/2 edge slots)
  * node masks: graph g only uses the slots inside its own n_g x n_g block. Noise, targets and
    the loss are restricted to that block, and generation samples n from the training sizes
    (as DiGress/DeFoG do). With a single graph size (Planar) the mask is all ones and everything
    reduces to the original method.
  * a node ordering applied before padding, since the per-slot prior gives every slot its own
    edge probability: 'bfs' (default), 'degree', or 'none' (as stored)
  * prior 'slot' (per-slot marginal, the original) or 'global' (one edge density for all slots)

Path:      x_t = 1[u < (1-t) * p0 + t * x1], with u shared across t, inside the node mask
Model:     GCNDiscrete, identical layers/parameter names to tacm_v3.GCN_Discrete
Loss:      BCE(student(x_t, t), sigmoid(EMA(x_t', t'))) + lam * BCE(student(x_t, t), x1),
           t' = max(t - delta, 0), delta ~ U[0.1, 0.3], averaged over in-mask slots
Coupling:  Hungarian match within the batch on the L1 distance between sorted degree sequences
           of the noise graph (drawn with the candidate target's mask) and the target graph
Sampling:  1..K steps of predict -> re-noise -> predict; re-noise either the predicted
           probabilities ('soft', original) or one concrete predicted graph ('hard'); steps can be
           packed toward t = 1 ('polyP' schedules); the last prediction becomes edges by
           threshold 0.5, Bernoulli sampling, or top-k
Backbone:  any module called as model(x_t, t, n_nodes) -> (B, D) logits: GCNDiscrete here, or
           defog_nn.SlotGraphTransformer (DeFoG's graph transformer)
"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ------------------------------------------------------------------
#  data: ordering, padding, masks
# ------------------------------------------------------------------

def order_nodes(G, how="bfs"):
    if how == "none":
        return list(G.nodes())
    deg = dict(G.degree())
    by_degree = sorted(G.nodes(), key=lambda v: (-deg[v], v))
    if how == "degree":
        return by_degree
    if how == "bfs":
        # BFS from the highest-degree node, visiting neighbours by descending degree;
        # restart from the highest-degree unvisited node for disconnected graphs.
        order, seen = [], set()
        for start in by_degree:
            if start in seen:
                continue
            seen.add(start)
            queue = [start]
            while queue:
                v = queue.pop(0)
                order.append(v)
                for w in sorted(G.neighbors(v), key=lambda w: (-deg[w], w)):
                    if w not in seen:
                        seen.add(w)
                        queue.append(w)
        return order
    raise ValueError(how)


class SlotLayout:
    """Maps between (B, N, N) adjacency matrices and (B, D) upper-triangle slot vectors."""

    def __init__(self, n_max, device):
        self.n = n_max
        self.idx = torch.triu_indices(n_max, n_max, offset=1).to(device)
        self.d = self.idx.shape[1]
        self.device = device
        inc = torch.zeros(self.d, n_max, device=device)      # slot -> its two endpoints
        inc[torch.arange(self.d), self.idx[0]] = 1.0
        inc[torch.arange(self.d), self.idx[1]] = 1.0
        self.incidence = inc

    def flatten(self, A):
        return A[:, self.idx[0], self.idx[1]]

    def unflatten(self, x):
        A = torch.zeros(x.shape[0], self.n, self.n, device=x.device)
        A[:, self.idx[0], self.idx[1]] = x
        return A + A.transpose(1, 2)

    def mask(self, n_nodes):
        """(B,) node counts -> (B, D) float mask of the slots inside each n x n block."""
        n_nodes = torch.as_tensor(n_nodes, device=self.device).view(-1, 1)
        return (self.idx[1].unsqueeze(0) < n_nodes).float()     # idx[0] < idx[1], so j < n suffices

    def degrees(self, x):
        return x @ self.incidence


def graphs_to_slots(graphs, layout, ordering):
    A = np.zeros((len(graphs), layout.n, layout.n), dtype=np.float32)
    n_nodes = []
    for k, G in enumerate(graphs):
        a = nx.to_numpy_array(G, nodelist=order_nodes(G, ordering))
        A[k, :len(a), :len(a)] = a
        n_nodes.append(len(a))
    x = layout.flatten(torch.tensor(A, device=layout.device))
    return x, torch.tensor(n_nodes, device=layout.device)


def slots_to_graphs(x, n_nodes, layout):
    out = []
    for a, n in zip(layout.unflatten(x).cpu().numpy(), n_nodes.tolist()):
        out.append(nx.from_numpy_array(a[:n, :n].astype(np.int8)))
    return out


def edge_prior(x1, mask, kind="slot"):
    if kind == "slot":
        return x1.sum(0) / mask.sum(0).clamp(min=1.0)
    if kind == "global":
        return torch.full_like(x1[0], float(x1.sum() / mask.sum()))
    raise ValueError(kind)


# ------------------------------------------------------------------
#  model (same layers and parameter names as tacm_v3.GCN_Discrete)
# ------------------------------------------------------------------

class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(np.log(10000) / (half - 1)))
        args = t * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, h, adj):
        n = h.shape[1]
        adj_hat = adj + torch.eye(n, device=adj.device).unsqueeze(0)
        deg = adj_hat.sum(-1, keepdim=True).clamp(min=1.0)
        h_out = self.linear(torch.bmm(adj_hat / deg, h)) + self.skip(h)
        return self.act(self.norm(h_out))


class GCNDiscrete(nn.Module):
    def __init__(self, n, time_dim=32, node_hidden=128, num_gcn_layers=4):
        super().__init__()
        self.n = n
        self.register_buffer("idx", torch.triu_indices(n, n, offset=1), persistent=False)
        self.time_mlp = nn.Sequential(SinusoidalTimeEmbeddings(time_dim), nn.Linear(time_dim, time_dim), nn.GELU())
        self.node_input = nn.Linear(n + time_dim, node_hidden)
        self.gcn_layers = nn.ModuleList([GCNLayer(node_hidden, node_hidden) for _ in range(num_gcn_layers)])
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_hidden, 2 * node_hidden), nn.GELU(),
            nn.Linear(2 * node_hidden, node_hidden), nn.GELU(),
            nn.Linear(node_hidden, 1))

    def forward(self, x, t, n_nodes=None):          # n_nodes unused: padding rows are already zero
        i, j = self.idx
        A = torch.zeros(x.shape[0], self.n, self.n, device=x.device)
        A[:, i, j] = x
        A = A + A.transpose(1, 2)
        t_emb = self.time_mlp(t).unsqueeze(1).expand(-1, self.n, -1)
        h = self.node_input(torch.cat([A, t_emb], dim=-1))
        for layer in self.gcn_layers:
            h = layer(h, A)
        return self.edge_mlp(torch.cat([h[:, i], h[:, j]], dim=-1)).squeeze(-1)


# ------------------------------------------------------------------
#  path, coupling, loss, sampling
# ------------------------------------------------------------------

def sample_path(u, p0, x1, t):
    """p0, x1 already masked. x_t = 1[u < (1-t) p0 + t x1]: a binary graph at every t."""
    return (u < (1 - t) * p0 + t * x1).float()


def couple(u, x1, mask, p_slot, layout):
    """Permutation of the batch that puts each noise draw u_a next to a target with a similar
    sorted degree sequence. Noise for candidate pair (a, b) is 1[u_a < p_slot * mask_b]."""
    B = x1.shape[0]
    x0 = (u.unsqueeze(1) < (p_slot.unsqueeze(0) * mask).unsqueeze(0)).float()   # (B, B, D)
    f0 = torch.sort(layout.degrees(x0.view(B * B, -1)), dim=1).values.view(B, B, -1)
    f1 = torch.sort(layout.degrees(x1), dim=1).values                           # (B, N)
    cost = (f0 - f1.unsqueeze(0)).abs().sum(-1).cpu().numpy()                    # cost[a, b]
    _, col = linear_sum_assignment(cost)
    return torch.as_tensor(col, device=x1.device)


def masked_bce(logits, target, mask):
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / mask.sum()


def rowwise_bce(logits, target, mask):
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum(1) / mask.sum(1).clamp(min=1.0)


def batch_loss(student, ema, x1, n_nodes, p_slot, layout, coupling=True, lam=3.0, target="prev",
               self_correct=0.0, sc_start=(0.3, 0.95), sc_jump=(0.4, 0.9)):
    """target='prev': EMA target at the noisier time t - delta (the original TACFM-D).
    target='next': EMA target at t + delta, closer to the data, as in Song et al.'s consistency
    models; where t + delta reaches 1 the target is the real graph (boundary condition).

    self_correct (sampler-aware training): the fraction of the batch trained on the input the
    sampler actually produces. For those rows:
        s ~ U[sc_start];  x_s = real graph noised to s;  guess = top-k of EMA(x_s, s)   (no grad)
        t = s + (1 - s) * U[sc_jump];  x_t = the GUESS noised to t
        loss = lam * BCE(student(x_t, t), real graph)          (no consistency term)
    so the model practises removing its own mistakes instead of only random noise. With
    self_correct=0 the computation is exactly the original one."""
    B = x1.shape[0]
    u = torch.rand(B, layout.d, device=x1.device)
    t = torch.rand(B, 1, device=x1.device)
    delta = torch.rand(B, 1, device=x1.device) * 0.2 + 0.1
    mask = layout.mask(n_nodes)
    if coupling:
        col = couple(u, x1, mask, p_slot, layout)
        x1, mask, n_nodes = x1[col], mask[col], n_nodes[col]
    p0 = p_slot.unsqueeze(0) * mask

    if self_correct <= 0:
        t_tgt = (t - delta).clamp(min=0.0) if target == "prev" else (t + delta).clamp(max=1.0)
        x_t = sample_path(u, p0, x1, t)
        x_tgt = sample_path(u, p0, x1, t_tgt)

        logits = student(x_t, t, n_nodes)
        with torch.no_grad():
            soft = torch.sigmoid(ema(x_tgt, t_tgt, n_nodes))
            if target == "next":
                soft = torch.where(t_tgt >= 1.0, x1, soft)
        return masked_bce(logits, soft, mask) + lam * masked_bce(logits, x1, mask)

    # ---- sampler-aware rows ----
    dev = x1.device
    sc = torch.rand(B, device=dev) < self_correct
    s = sc_start[0] + (sc_start[1] - sc_start[0]) * torch.rand(B, 1, device=dev)
    t_sc = s + (1 - s) * (sc_jump[0] + (sc_jump[1] - sc_jump[0]) * torch.rand(B, 1, device=dev))
    t = torch.where(sc.unsqueeze(1), t_sc, t)
    with torch.no_grad():
        x_s = sample_path(torch.rand_like(x1), p0, x1, s)
        guess = top_k_slots(torch.sigmoid(ema(x_s, s, n_nodes)) * mask, x1.sum(1).long(), mask)
    source = torch.where(sc.unsqueeze(1), guess, x1)       # what gets noised to time t

    t_tgt = (t - delta).clamp(min=0.0) if target == "prev" else (t + delta).clamp(max=1.0)
    x_t = sample_path(u, p0, source, t)
    x_tgt = sample_path(u, p0, x1, t_tgt)

    logits = student(x_t, t, n_nodes)
    with torch.no_grad():
        soft = torch.sigmoid(ema(x_tgt, t_tgt, n_nodes))
        if target == "next":
            soft = torch.where(t_tgt >= 1.0, x1, soft)
    consistency = rowwise_bce(logits, soft, mask) * (~sc).float()   # only for the ordinary rows
    denoise = rowwise_bce(logits, x1, mask)
    return (consistency + lam * denoise).mean()


def top_k_slots(scores, k, mask):
    """Binary (B, D): the k[b] highest-scoring in-mask slots of each row."""
    scores = scores.masked_fill(mask == 0, -1.0)
    order = torch.argsort(scores, dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(scores.shape[1], device=scores.device).expand_as(order))
    return (ranks < k.view(-1, 1)).float() * mask


def step_times(steps, schedule="uniform"):
    """Sampling times in [0, 1). 'uniform': 0, 1/K, 2/K, ...; 'polyP': 1 - (1 - k/K)^P, which packs
    the steps toward t = 1, where the last few wrong edges get fixed (like DeFoG's polydec)."""
    u = torch.linspace(0.0, 1.0, steps + 1)[:-1]
    if schedule == "uniform":
        return u.tolist()
    if schedule.startswith("poly"):
        return (1 - (1 - u) ** float(schedule[4:])).tolist()
    raise ValueError(schedule)


def _to_edges(A_hat, final, n_edges, mask):
    if final == "threshold":
        return (A_hat > 0.5).float()
    if final == "sample":
        return (torch.rand_like(A_hat) < A_hat).float()
    if final == "topk":
        return top_k_slots(A_hat, n_edges, mask)
    raise ValueError(final)


@torch.no_grad()
def generate(model, p_slot, n_nodes, layout, steps, final="threshold", n_edges=None, times=None,
             renoise="soft", schedule="uniform"):
    """n_nodes: (B,) node counts of the graphs to generate.
    final:    how a prediction becomes edges -- 'threshold' (p > 0.5, the original),
              'sample' (Bernoulli(p)), or 'topk' (the n_edges[b] most likely slots).
    renoise:  what is re-noised between steps -- 'soft': the predicted probabilities (the original);
              'hard': one concrete graph, made from the prediction with the `final` rule. 'hard'
              matches training, where x_t is always a noised version of a single real graph.
    times / schedule: explicit step times in [0, 1), or a schedule name for step_times()."""
    model.eval()
    mask = layout.mask(n_nodes)
    p0 = p_slot.unsqueeze(0) * mask
    x = (torch.rand_like(p0) < p0).float()
    ts = list(times) if times is not None else step_times(steps, schedule)
    for k, tv in enumerate(ts):
        A_hat = torch.sigmoid(model(x, torch.full((x.shape[0], 1), float(tv), device=x.device), n_nodes)) * mask
        if k == len(ts) - 1:
            x = _to_edges(A_hat, final, n_edges, mask)
        else:
            t_next = float(ts[k + 1])
            target = _to_edges(A_hat, final, n_edges, mask) if renoise == "hard" else A_hat
            x = (torch.rand_like(A_hat) < (1 - t_next) * p0 + t_next * target).float()
    model.train()
    return x
