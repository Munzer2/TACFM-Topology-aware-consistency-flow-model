# Vendored from DeFoG (github.com/manuelmlmadeira/DeFoG, src/models/transformer_model.py and
# src/models/layers.py; MIT licence, (c) Manuel Madeira, Yiming Qin), itself based on DiGress.
# Changes: the src.utils / flow_matching imports are replaced by the inline `_mask` helper, and the
# debug-only masking asserts are dropped. Layers, initialisation and forward pass are unchanged.

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm


def _mask(X, E, node_mask):
    x_mask = node_mask.unsqueeze(-1)
    e_mask1 = x_mask.unsqueeze(2)
    e_mask2 = x_mask.unsqueeze(1)
    return X * x_mask, E * e_mask1 * e_mask2


class Xtoy(nn.Module):
    def __init__(self, dx, dy):
        """Map node features to global features"""
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X):
        m = X.mean(dim=1)
        mi = X.min(dim=1)[0]
        ma = X.max(dim=1)[0]
        std = X.std(dim=1)
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


class Etoy(nn.Module):
    def __init__(self, d, dy):
        """Map edge features to global features."""
        super().__init__()
        self.lin = nn.Linear(4 * d, dy)

    def forward(self, E):
        m = E.mean(dim=(1, 2))
        mi = E.min(dim=2)[0].min(dim=1)[0]
        ma = E.max(dim=2)[0].max(dim=1)[0]
        std = torch.std(E, dim=(1, 2))
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


def masked_softmax(x, mask, **kwargs):
    if mask.sum() == 0:
        return x
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")
    return torch.softmax(x_masked, **kwargs)


class XEyTransformerLayer(nn.Module):
    """Transformer layer that updates node (X), edge (E) and global (y) features."""

    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048, dim_ffE: int = 128,
                 dim_ffy: int = 2048, dropout: float = 0.1, layer_norm_eps: float = 1e-5,
                 device=None, dtype=None) -> None:
        kw = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        X = self.normX1(X + self.dropoutX1(newX))
        E = self.normE1(E + self.dropoutE1(newE))
        y = self.norm_y1(y + self.dropout_y1(new_y))

        ff_outputX = self.dropoutX3(self.linX2(self.dropoutX2(self.activation(self.linX1(X)))))
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.dropoutE3(self.linE2(self.dropoutE2(self.activation(self.linE1(E)))))
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.dropout_y3(self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y)))))
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """Self attention layer that also updates the representations on the edges."""

    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dropout_attn = Dropout(0.1)
        self.dropout_X = Dropout(0.1)
        self.dropout_E = Dropout(0.1)

        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask
        K = self.k(X) * x_mask
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))
        Q = Q.unsqueeze(2)  # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)  # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))

        # Incorporate edge features to the self attention scores.
        E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        E2 = self.e_add(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)  # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2  # bs, n, n, de

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)
        attn = masked_softmax(Y, softmax_mask, dim=2)

        V = self.v(X) * x_mask
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)  # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = (attn * V).sum(dim=2)
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask

        # Process y based on X and E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = self.y_out(y + x_y + e_y)  # bs, dy

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]), act_fn_in)
        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]), act_fn_in)
        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"] + 64, hidden_mlp_dims["y"]), act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]), act_fn_in)

        self.tf_layers = nn.ModuleList([
            XEyTransformerLayer(dx=hidden_dims["dx"], de=hidden_dims["de"], dy=hidden_dims["dy"],
                                n_head=hidden_dims["n_head"], dim_ffX=hidden_dims["dim_ffX"],
                                dim_ffE=hidden_dims["dim_ffE"])
            for _ in range(n_layers)])

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]))
        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]))
        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]), act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], output_dims["y"]))

    def forward(self, X, E, y, node_mask):
        """X (bs, n, dX), E (bs, n, n, dE), y (bs, dy) with the time t as its LAST column."""
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2

        # encode time steps additionally
        time_emb = timestep_embedding(y[:, -1].unsqueeze(-1), 64)
        y = torch.hstack([y, time_emb])

        X, E = _mask(self.mlp_in_X(X), new_E, node_mask)
        y = self.mlp_in_y(y)

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = X + X_to_out
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1 / 2 * (E + torch.transpose(E, 1, 2))
        X, E = _mask(X, E, node_mask)
        return X, E, y


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal timestep embeddings; timesteps (N, 1), may be fractional."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
                      ).to(device=timesteps.device)
    args = timesteps.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
