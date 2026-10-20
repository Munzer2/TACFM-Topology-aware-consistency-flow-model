"""
DeFoG's graph transformer (and DiGress/DeFoG extra features) as a TACFM-D backbone.

The network is vendored unchanged (MIT licence, see graph_transformer.py / extra_features.py).
SlotGraphTransformer only adapts the interface to tacfm_d: it takes the binary slot vector x_t,
the time t and the node counts, builds the same inputs DeFoG builds (one-hot node/edge states,
extra features, t as the last global feature), and returns one edge logit per slot:
logit = E_out[..., edge] - E_out[..., no-edge], so sigmoid(logit) equals DeFoG's softmax edge
probability and the BCE losses in tacfm_d apply unchanged.
"""

import torch
import torch.nn as nn

from .extra_features import ExtraFeatures
from .graph_transformer import GraphTransformer

# DeFoG's configs/experiment/{planar,sbm}.yaml (model section)
PRESETS = {
    "planar": dict(n_layers=10, features="rrwp", rrwp_steps=12,
                   hidden_mlp_dims={"X": 128, "E": 64, "y": 128},
                   hidden_dims={"dx": 256, "de": 64, "dy": 64, "n_head": 8,
                                "dim_ffX": 256, "dim_ffE": 64, "dim_ffy": 256}),
    "sbm": dict(n_layers=8, features="rrwp", rrwp_steps=20,
                hidden_mlp_dims={"X": 128, "E": 64, "y": 128},
                hidden_dims={"dx": 256, "de": 64, "dy": 64, "n_head": 8,
                             "dim_ffX": 256, "dim_ffE": 64, "dim_ffy": 256}),
}


class SlotGraphTransformer(nn.Module):
    def __init__(self, n_max, n_layers, hidden_mlp_dims, hidden_dims, features="rrwp", rrwp_steps=12,
                 amp=False):
        super().__init__()
        self.n = n_max
        self.amp = amp
        self.register_buffer("idx", torch.triu_indices(n_max, n_max, offset=1), persistent=False)
        self.extra = ExtraFeatures(features, rrwp_steps, n_max)
        dX, dE, dy = self.extra.dims()
        self.net = GraphTransformer(
            n_layers=n_layers,
            input_dims={"X": 1 + dX, "E": 2 + dE, "y": dy + 1},
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims={"X": 1, "E": 2, "y": 0},
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

    def forward(self, x, t, n_nodes):
        B, n = x.shape[0], self.n
        i, j = self.idx
        node_mask = torch.arange(n, device=x.device).unsqueeze(0) < n_nodes.view(-1, 1)
        pair = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        off_diag = pair & ~torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0)

        A = torch.zeros(B, n, n, device=x.device)
        A[:, i, j] = x.float()
        A = (A + A.transpose(1, 2)) * pair

        with torch.autocast(device_type=x.device.type, enabled=False):
            fX, fE, fy = self.extra(A, node_mask)       # float32: cycle counts need exact arithmetic

        E = torch.stack([1 - A, A], dim=-1) * off_diag.unsqueeze(-1)   # one-hot; diagonal and padding all-zero
        X = node_mask.float().unsqueeze(-1)                             # a single node type
        X = torch.cat([X, fX], dim=-1)
        E = torch.cat([E, fE], dim=-1)
        y = torch.cat([fy, t.float().view(B, 1)], dim=-1)               # t must be the last column

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            _, E_out, _ = self.net(X, E, y, node_mask)
        logits = (E_out[..., 1] - E_out[..., 0]).float()
        return logits[:, i, j]
