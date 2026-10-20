"""Builds Codes/SBM/notebooks/discrete_pipeline_walkthrough.ipynb (unexecuted).

Run from anywhere:  python Codes/SBM/notebooks/build_discrete_walkthrough.py
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code

cells = []

# ======================================================================
cells.append(md(r"""# TACFM-D walkthrough: the discrete consistency model, step by step

This notebook rebuilds the **current discrete version** of the model one piece at a time, so you can see what each part does.
It uses the same code as `tacm_v3.py` (path, model, loss, sampler) and `coupling_experiment.py` (topology-aware coupling). Cell E checks that the model here is identical to the one in `tacm_v3.py`.

**Dataset:** Community_small (the corrected v2 version: 400 train / 100 held-out test graphs, 12–20 nodes).
Later we will swap in **Planar** and **SBM** from the SPECTRE benchmark. Part K lists what has to change for that.

| Part | What it shows |
|---|---|
| A | Setup |
| B | The data, and how a graph becomes a vector of 0/1 edge slots |
| C | The noise: the per-slot edge prior `p_marg` |
| D | **The discrete path**: how noise turns into a graph, with one shared random draw `u` |
| E | The model: a GCN that predicts the final graph |
| F | **Topology-aware coupling**: pairing noise with similar real graphs |
| G | One training step, by hand |
| H | The training loop |
| I | Generating graphs in 1 or 4 steps |
| J | Scoring with MMD against held-out graphs |
| K | What changes for Planar + SBM (SPECTRE) |

**The whole pipeline in one picture**

```
TRAINING                                            GENERATION
real graph x1 ──┐                                   random graph x ~ Bernoulli(p_marg)
noise u ────────┼─> coupling (pair by degrees)            │
time t, t-δ ────┘          │                              ▼
                           ▼                     ┌─> model predicts final graph Â
             x_t, x_(t-δ) on the same path       │         │
                           │                     │   last step? ── yes ──> output 1[Â > 0.5]
          student(x_t) vs EMA(x_(t-δ))  + vs x1  │         │ no
                     BCE loss                    └── re-noise toward Â, next time
```

Running everything takes about 3 minutes on a CPU with `EPOCHS = 200` (about 7 minutes at 600). The report's numbers use 600 epochs and 3 seeds.
"""))

# ======================================================================
cells.append(md(r"""## A. Setup"""))

cells.append(code(r"""import sys, time, copy, pickle
from pathlib import Path

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# Find Codes/SBM (the folder holding metrics_v2.py), whether this runs from notebooks/ or from SBM/.
HERE = Path.cwd()
SBM_DIR = next(p for p in [HERE, HERE.parent, HERE / "Codes" / "SBM"] if (p / "metrics_v2.py").exists())
sys.path.insert(0, str(SBM_DIR))
from metrics_v2 import degree_stats, clustering_stats, spectral_stats   # the corrected MMD metrics

SEED       = 0
EPOCHS     = 200      # the report uses 600
BATCH_SIZE = 16
LR         = 5e-4
EMA_RATE   = 0.995    # how slowly the target copy follows the student
LAMBDA     = 3.0      # weight of the "match the real graph" loss term
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
print("device:", DEVICE, "| code folder:", SBM_DIR)
"""))

# ======================================================================
cells.append(md(r"""## B. The data

Each graph is stored as a networkx graph. The model needs a fixed-size input, so every graph is:
1. turned into an adjacency matrix and **zero-padded** to `N_MAX × N_MAX` (here 20 × 20),
2. reduced to its **upper triangle** (the matrix is symmetric, so the lower half is redundant). That gives `N_MAX·(N_MAX−1)/2` = **190 edge slots**, each 0 or 1.

So a graph is just a vector of 190 zeros and ones. Slot `k` always means the same node pair `(i, j)`.
"""))

cells.append(code(r"""train_graphs = pickle.load(open(SBM_DIR / "data/community_v2_train.pkl", "rb"))
test_graphs  = pickle.load(open(SBM_DIR / "data/community_v2_test.pkl", "rb"))
N_MAX = max(G.number_of_nodes() for G in train_graphs + test_graphs)

def to_dense(graphs, n):
    # Node order = the order networkx stores the nodes in. Slot meanings depend on it (see Part K).
    A = np.zeros((len(graphs), n, n), dtype=np.float32)
    for k, G in enumerate(graphs):
        a = nx.to_numpy_array(G)
        A[k, :len(a), :len(a)] = a
    return torch.tensor(A, device=DEVICE)

IDX = torch.triu_indices(N_MAX, N_MAX, offset=1).to(DEVICE)   # slot k <-> node pair (IDX[0,k], IDX[1,k])
D = IDX.shape[1]                                              # number of edge slots

def flatten(A):            # (B, N, N) -> (B, D)
    return A[:, IDX[0], IDX[1]]

def unflatten(x):          # (B, D) -> (B, N, N), symmetric
    A = torch.zeros(x.shape[0], N_MAX, N_MAX, device=x.device)
    A[:, IDX[0], IDX[1]] = x
    return A + A.transpose(1, 2)

A_train = to_dense(train_graphs, N_MAX)
x1_train = flatten(A_train)

print(f"train {len(train_graphs)} graphs, test {len(test_graphs)} graphs, N_MAX = {N_MAX}, slots D = {D}")
print(f"test set: {np.mean([g.number_of_nodes() for g in test_graphs]):.1f} nodes, "
      f"{np.mean([g.number_of_edges() for g in test_graphs]):.1f} edges on average")
print("first 5 slots are node pairs:", [(int(i), int(j)) for i, j in zip(IDX[0, :5], IDX[1, :5])])
print("graph 0 as a vector (first 40 slots):", x1_train[0, :40].int().tolist())
assert torch.equal(unflatten(x1_train[:3]), A_train[:3])   # flatten/unflatten round-trip
"""))

cells.append(code(r"""fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for k, ax in enumerate(axes[:3]):
    G = train_graphs[k]
    nx.draw(G, pos=nx.spring_layout(G, seed=0), ax=ax, node_size=40, edge_color="gray")
    ax.set_title(f"train graph {k}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
axes[3].imshow(A_train[0].cpu(), cmap="Greys")
axes[3].set_title("graph 0, padded to 20 x 20")
plt.tight_layout(); plt.show()
"""))

# ======================================================================
cells.append(md(r"""## C. The noise: a per-slot edge prior

Instead of Gaussian noise, the starting point is a **random graph**. Each slot is switched on independently with probability

`p_marg[k]` = how often slot `k` is an edge in the training set.

Because graphs are padded, slots that touch high-numbered nodes (18, 19) are almost always empty, and `p_marg` already knows that. So a noise graph has a realistic number of nodes and edges before the model has done anything.

This "one probability per slot" prior is the part no other graph paper uses. It only makes sense because slot `k` always means the same node pair (see Part K).
"""))

cells.append(code(r"""p_marg = x1_train.mean(0)                     # (D,)

x0_example = (torch.rand(D, device=DEVICE) < p_marg).float()
G0 = nx.from_numpy_array(unflatten(x0_example[None])[0].cpu().numpy())
G0.remove_nodes_from(list(nx.isolates(G0)))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
im = axes[0].imshow(unflatten(p_marg[None])[0].cpu(), cmap="viridis")
axes[0].set_title("p_marg as a 20 x 20 matrix"); plt.colorbar(im, ax=axes[0])
nx.draw(G0, pos=nx.spring_layout(G0, seed=0), ax=axes[1], node_size=40, edge_color="gray")
axes[1].set_title(f"one noise graph: {G0.number_of_nodes()} nodes, {G0.number_of_edges()} edges")
plt.tight_layout(); plt.show()
print(f"p_marg: mean {p_marg.mean():.3f}, min {p_marg.min():.3f}, max {p_marg.max():.3f}")
"""))

# ======================================================================
cells.append(md(r"""## D. The discrete path

For a real graph `x1`, draw **one** uniform vector `u` (one number in [0, 1] per slot) and keep it fixed. For any time `t` in [0, 1]:

$$p_t = (1-t)\,p_\text{marg} + t\,x_1, \qquad x_t = \mathbb{1}[\,u < p_t\,]$$

- At `t = 0`: `x_0` is a noise graph drawn from `p_marg`.
- At `t = 1`: `p_1 = x1`, so `x_1` is exactly the real graph.
- In between: `x_t` is always a **real 0/1 graph**, never a blurry in-between.

Because `u` is shared, each slot changes **at most once**. A slot that should become an edge switches on at the moment `p_t` passes `u`, and stays on. That is what makes "two points on the same path" meaningful for consistency training.

The probability `p_t` is the standard linear path from discrete flow matching (DeFoG). The shared `u` is the two-state version of the shared-noise trick in DUO and MultiMDM.
"""))

cells.append(code(r"""def sample_path(u, p0, x1, t):
    # x_t = 1[u < (1-t) p0 + t x1]  -- a real binary graph at every t
    p_t = (1 - t) * p0.unsqueeze(0) + t * x1
    return (u < p_t).float()

x1 = x1_train[:1]
u = torch.rand(1, D, device=DEVICE)
ts = [0.0, 0.25, 0.5, 0.75, 1.0]

fig, axes = plt.subplots(1, len(ts), figsize=(15, 3.2))
for ax, t in zip(axes, ts):
    xt = sample_path(u, p_marg, x1, torch.tensor([[t]], device=DEVICE))
    ax.imshow(unflatten(xt)[0].cpu(), cmap="Greys")
    ax.set_title(f"t = {t}: {int(xt.sum())} edges")
    ax.axis("off")
plt.suptitle("one path from noise (t=0) to the real graph (t=1)"); plt.tight_layout(); plt.show()

# Check: on a fine time grid, no slot changes more than once.
grid = torch.linspace(0, 1, 201, device=DEVICE)[:, None]
traj = torch.cat([sample_path(u, p_marg, x1, t[None]) for t in grid])     # (201, D)
flips = (traj[1:] != traj[:-1]).sum(0)
print("max number of flips of any slot along the path:", int(flips.max()))
print("x at t=1 equals the real graph:", bool(torch.equal(traj[-1], x1[0])))
"""))

# ======================================================================
cells.append(md(r"""## E. The model: a GCN that predicts the final graph

The network takes the current graph `x_t` and the time `t`, and outputs **one logit per slot**: how likely that edge is in the *final* graph. (It does not predict a velocity; this is a consistency model.)

Inside:
1. Rebuild the `N × N` adjacency matrix from the 190-slot vector.
2. Each node starts with its own adjacency row plus a sinusoidal embedding of `t`, projected to 128 numbers.
3. **Four GCN layers.** Each node averages its neighbours' vectors (plus its own, through a self-loop), applies a linear layer, adds a skip connection, then LayerNorm and GELU. Since `x_t` is a real 0/1 graph, messages travel along real edges.
4. For every node pair `(i, j)`, concatenate the two node vectors and pass them through a small MLP. That gives the logit for edge `(i, j)`.
"""))

cells.append(code(r"""class SinusoidalTimeEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):                                   # t: (B, 1)
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(np.log(10000) / (half - 1)))
        args = t * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()

    def forward(self, h, adj):
        n = h.shape[1]
        adj_hat = adj + torch.eye(n, device=adj.device).unsqueeze(0)      # self-loops
        deg = adj_hat.sum(-1, keepdim=True).clamp(min=1.0)
        h_out = self.linear(torch.bmm(adj_hat / deg, h)) + self.skip(h)   # average neighbours, transform, skip
        return self.act(self.norm(h_out))


class GCNDiscrete(nn.Module):
    # Same layers and parameter names as tacm_v3.GCN_Discrete, so weights are interchangeable.
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

    def forward(self, x, t):                       # x: (B, D) binary, t: (B, 1)
        i, j = self.idx
        A = torch.zeros(x.shape[0], self.n, self.n, device=x.device)
        A[:, i, j] = x
        A = A + A.transpose(1, 2)                  # (B, N, N)
        t_emb = self.time_mlp(t).unsqueeze(1).expand(-1, self.n, -1)
        h = self.node_input(torch.cat([A, t_emb], dim=-1))          # node features: adjacency row + time
        for layer in self.gcn_layers:
            h = layer(h, A)                                         # message passing on the real graph
        return self.edge_mlp(torch.cat([h[:, i], h[:, j]], dim=-1)).squeeze(-1)   # (B, D) logits


model = GCNDiscrete(N_MAX).to(DEVICE)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
with torch.no_grad():
    out = model(x1_train[:4], torch.rand(4, 1, device=DEVICE))
print("input", tuple(x1_train[:4].shape), "-> output logits", tuple(out.shape))
"""))

cells.append(code(r"""# Check: this model is the same as the one in tacm_v3.py (same weights -> same outputs).
if N_MAX == 20:
    from tacm_v3 import GCN_Discrete as ReferenceModel
    ref = ReferenceModel(N_MAX).to(DEVICE)
    ours = GCNDiscrete(N_MAX).to(DEVICE)
    ours.load_state_dict(ref.state_dict())
    x, t = x1_train[:8], torch.rand(8, 1, device=DEVICE)
    with torch.no_grad():
        print("identical to tacm_v3.GCN_Discrete:", torch.allclose(ours(x, t), ref(x, t), atol=1e-6))
"""))

# ======================================================================
cells.append(md(r"""## F. Topology-aware coupling

Normally each noise graph is paired with a **random** real graph. At `t = 0` the noise then tells the model nothing about which graph it should become, so 1-step generation has to guess.

Topology-aware coupling re-pairs them inside each batch:
1. Compute the **sorted degree sequence** of each noise graph `x0 = 1[u < p_marg]` and of each real graph.
2. Cost of pairing noise `a` with real graph `b` = L1 distance between their sorted degree sequences.
3. The **Hungarian algorithm** finds the pairing with the lowest total cost.

Only the pairing changes. The noise and the data are the same as before, and no extra randomness is used.
"""))

cells.append(code(r"""# B_inc[k, i] = 1 if slot k touches node i, so x @ B_inc gives every node's degree.
B_inc = torch.zeros(D, N_MAX, device=DEVICE)
B_inc[torch.arange(D), IDX[0]] = 1.0
B_inc[torch.arange(D), IDX[1]] = 1.0

def couple(u, x1, p0):
    x0 = (u < p0.unsqueeze(0)).float()
    f0 = torch.sort(x0 @ B_inc, dim=1).values        # sorted degrees of the noise graphs
    f1 = torch.sort(x1 @ B_inc, dim=1).values        # sorted degrees of the real graphs
    cost = torch.cdist(f0, f1, p=1).cpu().numpy()
    _, col = linear_sum_assignment(cost)
    return x1[torch.as_tensor(col, device=x1.device)], cost, col

xb = x1_train[torch.randperm(len(x1_train))[:BATCH_SIZE]]
ub = torch.rand(BATCH_SIZE, D, device=DEVICE)
xb_paired, cost, col = couple(ub, xb, p_marg)
print(f"average degree-sequence distance, random pairing:   {np.mean(np.diag(cost)):.1f}")
print(f"average degree-sequence distance, after Hungarian:  {np.mean(cost[np.arange(BATCH_SIZE), col]):.1f}")

x0b = (ub < p_marg).float()
plt.figure(figsize=(6, 3))
plt.plot(torch.sort(x0b[0] @ B_inc).values.cpu(), "o-", label="noise graph 0")
plt.plot(torch.sort(xb[0] @ B_inc).values.cpu(), "s--", label="random partner")
plt.plot(torch.sort(xb_paired[0] @ B_inc).values.cpu(), "^-", label="matched partner")
plt.xlabel("node (sorted by degree)"); plt.ylabel("degree"); plt.legend(); plt.tight_layout(); plt.show()
"""))

# ======================================================================
cells.append(md(r"""## G. One training step, by hand

For each graph in the batch:
1. Draw `u`, a time `t ~ U[0, 1]` and a gap `δ ~ U[0.1, 0.3]`; set `t' = max(t − δ, 0)`.
2. (Optional) re-pair the batch with the coupling from Part F.
3. Build **two points on the same path**: `x_t` and `x_t'` (same `u`, same `x1`).
4. The **student** predicts the final graph from `x_t`.
5. The **EMA copy** (a slowly-updated copy of the student, no gradients) predicts the final graph from `x_t'`.
6. Loss = BCE(student, EMA's probabilities) + `λ` · BCE(student, real graph), with `λ = 3`.

The first term says "points on the same path should predict the same graph". The second anchors the prediction to the real data.

Note: here `t'` is the *noisier* point. The original consistency-model recipe takes the target from the point closer to the data. This is how the current code does it; testing the other direction is on the to-do list.
"""))

cells.append(code(r"""def batch_loss(student, ema, x1, p0, use_coupling, return_parts=False):
    B = x1.shape[0]
    u = torch.rand(B, D, device=DEVICE)
    t = torch.rand(B, 1, device=DEVICE)
    delta = torch.rand(B, 1, device=DEVICE) * 0.2 + 0.1
    t_prev = (t - delta).clamp(min=0.0)
    if use_coupling:
        x1 = couple(u, x1, p0)[0]

    x_t = sample_path(u, p0, x1, t)
    x_prev = sample_path(u, p0, x1, t_prev)

    logits = student(x_t, t)
    with torch.no_grad():
        target = torch.sigmoid(ema(x_prev, t_prev))
    consistency = F.binary_cross_entropy_with_logits(logits, target)
    denoise = F.binary_cross_entropy_with_logits(logits, x1)
    loss = consistency + LAMBDA * denoise
    if return_parts:
        return loss, consistency, denoise, x_t, x_prev
    return loss

student = GCNDiscrete(N_MAX).to(DEVICE)
ema = copy.deepcopy(student).eval()
for p in ema.parameters():
    p.requires_grad = False

loss, cons, den, x_t, x_prev = batch_loss(student, ema, xb, p_marg, use_coupling=True, return_parts=True)
print(f"edges in x_t (first 4 graphs):  {x_t[:4].sum(1).int().tolist()}")
print(f"edges in x_t' (first 4 graphs): {x_prev[:4].sum(1).int().tolist()}   <- noisier, earlier on the same paths")
print(f"consistency BCE {cons.item():.3f} + {LAMBDA} x denoising BCE {den.item():.3f} = loss {loss.item():.3f}")
"""))

# ======================================================================
cells.append(md(r"""## H. The training loop

Standard loop: Adam with a cosine learning-rate schedule, gradient clipping at 1.0, and after every step the EMA copy moves 0.5% toward the student:

`EMA ← 0.995 · EMA + 0.005 · student`

`USE_COUPLING = True` trains the topology-aware version (the best one in the report). Set it to `False` for the independent baseline.
"""))

cells.append(code(r"""USE_COUPLING = True

def train(epochs, use_coupling, seed=SEED):
    torch.manual_seed(seed)
    student = GCNDiscrete(N_MAX).to(DEVICE)
    ema = copy.deepcopy(student).eval()
    for p in ema.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    losses, start = [], time.time()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(x1_train), device=DEVICE)
        running = 0.0
        for k in range(0, len(perm), BATCH_SIZE):
            loss = batch_loss(student, ema, x1_train[perm[k:k + BATCH_SIZE]], p_marg, use_coupling)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            opt.step()
            with torch.no_grad():
                for pe, ps in zip(ema.parameters(), student.parameters()):
                    pe.mul_(EMA_RATE).add_(ps, alpha=1 - EMA_RATE)
            running += loss.item()
        sched.step()
        losses.append(running / ((len(perm) + BATCH_SIZE - 1) // BATCH_SIZE))
        if ep % 50 == 0 or ep == epochs:
            print(f"epoch {ep:4d}  loss {losses[-1]:.4f}  ({time.time() - start:.0f}s)")
    return student, losses

student, losses = train(EPOCHS, USE_COUPLING)
plt.figure(figsize=(6, 3)); plt.plot(losses); plt.xlabel("epoch"); plt.ylabel("loss"); plt.tight_layout(); plt.show()
"""))

# ======================================================================
cells.append(md(r"""## I. Generating graphs in 1 or 4 steps

No ODE solver is used. The model jumps straight to a guess of the final graph:

1. Start from a noise graph `x ~ Bernoulli(p_marg)` at time `τ = 0`.
2. The model predicts the final edge probabilities `Â`.
3. **Last step:** keep the edges with `Â > 0.5`. Done.
4. **Otherwise:** re-noise toward the next time `τ'`: draw `x ~ Bernoulli((1 − τ') · p_marg + τ' · Â)`, then go back to step 2.

With 4 steps the times are 0, 0.25, 0.5, 0.75. Each round the model sees a cleaner graph, so its guess improves.
"""))

cells.append(code(r"""@torch.no_grad()
def generate(model, p0, n, steps, return_chain=False):
    model.eval()
    x = (torch.rand(n, D, device=DEVICE) < p0.unsqueeze(0)).float()
    ts = torch.linspace(0.0, 1.0, steps + 1)[:-1]
    chain = [x]
    for k, tv in enumerate(ts):
        A_hat = torch.sigmoid(model(x, torch.full((n, 1), tv.item(), device=DEVICE)))
        if k == len(ts) - 1:
            x = (A_hat > 0.5).float()
        else:
            t_next = ts[k + 1].item()
            x = (torch.rand_like(A_hat) < (1 - t_next) * p0.unsqueeze(0) + t_next * A_hat).float()
        chain.append(x)
    model.train()
    return (x, chain) if return_chain else x

def to_graphs(x):
    # Same as evaluate.adj_to_graphs: isolated nodes (padding) are dropped.
    graphs = []
    for a in unflatten(x).cpu().numpy():
        G = nx.from_numpy_array(a)
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() > 0:
            graphs.append(G)
    return graphs

_, chain = generate(student, p_marg, 1, steps=4, return_chain=True)
fig, axes = plt.subplots(1, len(chain), figsize=(15, 3.2))
for k, (ax, xk) in enumerate(zip(axes, chain)):
    ax.imshow(unflatten(xk)[0].cpu(), cmap="Greys"); ax.axis("off")
    ax.set_title("start (noise)" if k == 0 else f"after step {k}: {int(xk.sum())} edges")
plt.suptitle("4-step generation of one graph"); plt.tight_layout(); plt.show()
"""))

cells.append(code(r"""samples = {steps: to_graphs(generate(student, p_marg, 100, steps)) for steps in (1, 4)}

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for row, steps in enumerate((1, 4)):
    for ax, G in zip(axes[row], samples[steps][:4]):
        nx.draw(G, pos=nx.spring_layout(G, seed=0), ax=ax, node_size=30, edge_color="gray")
        ax.set_title(f"{steps}-step: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
plt.tight_layout(); plt.show()
"""))

# ======================================================================
cells.append(md(r"""## J. Scoring

MMD compares the distribution of a graph statistic between generated graphs and the **held-out test graphs**. Lower is better; 0 means identical.

- **Deg.** node degrees, **Clus.** clustering coefficients (triangles), **Spec.** Laplacian eigenvalues.
- Degree and clustering use the same settings as the GDSS reference code, so they can be compared with published numbers.

The "floor" row scores real training graphs against the test graphs. That is about the best any model can do with this many samples. It can be slightly negative because the estimate is noisy.
"""))

cells.append(code(r"""def score(gen):
    return (degree_stats(test_graphs, gen), clustering_stats(test_graphs, gen), spectral_stats(test_graphs, gen))

rows = {"floor (train vs test)": score(train_graphs[:100])}
for steps in (1, 4):
    rows[f"this run, {steps}-step"] = score(samples[steps])

print(f"{'':28s}{'Deg.':>8s}{'Clus.':>8s}{'Spec.':>8s}")
for name, (d, c, s) in rows.items():
    print(f"{name:28s}{d:8.3f}{c:8.3f}{s:8.3f}")
print("\nreport (600 epochs, coupling, mean of 3 seeds), 4-step:  0.090   0.068   0.072")
print("GDSS (1000 steps, from its paper):                        0.045   0.086      --")
"""))

# ======================================================================
cells.append(md(r"""## K. What changes for Planar + SBM (SPECTRE)

| | Community_small (now) | Planar | SBM |
|---|---|---|---|
| Graphs | 500 (400 / 100) | 200 (128 train / 32 val / 40 test) | 200 (128 / 32 / 40) |
| Nodes | 12–20 | 64 (all the same) | 44–187 |
| Edge slots `D` | 190 | 2,016 | up to 17,391 |
| Validity check | – | graph is planar and connected | 2–5 communities, checked with a statistical test |
| Metrics | Deg / Clus / Spec MMD | Deg / Clus / Orbit / Spec / Wavelet MMD + **V.U.N.** (% valid, unique, novel) | same as Planar |

**Things that must change in the code:**
1. **Data.** Load the SPECTRE `.pt` files (the DiGress and DeFoG repos have them and their fixed splits) and set `N_MAX` from the data. Everything in Parts B–I already uses `N_MAX` and `D`, so the code here mostly carries over.
2. **Node ordering.** `p_marg` gives each *slot* its own probability, so slot `(3, 7)` must mean something similar in every graph. On Community_small the networkx order happens to work. For Planar and SBM we must choose an ordering (e.g. BFS from the highest-degree node, or sort by degree) and also try a single shared probability for all slots.
3. **Compute.** SBM's dense 187 × 187 graphs with 17k slots need a GPU. Planar (64 nodes) should be fine to try on a CPU first.
4. **Evaluation.** Use the DiGress/DeFoG evaluation code (it includes orbit counts through ORCA, wavelet MMD and V.U.N.), so our numbers are on exactly the same protocol as the papers we compare with.
5. **Baselines at few steps.** Run DeFoG at 1, 4 and 10 steps on the same data, so we can show quality vs number of steps.
"""))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).with_name("discrete_pipeline_walkthrough.ipynb")
nbformat.write(nb, out)
print("wrote", out)
