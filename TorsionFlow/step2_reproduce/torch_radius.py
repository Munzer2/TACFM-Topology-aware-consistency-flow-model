"""
torch_radius.py - plain-PyTorch replacements for torch_cluster.radius and
torch_cluster.radius_graph, as Torsional Diffusion uses them.

patch_windows.py copies this file into the repo as utils/torch_radius.py.

Same signatures and output convention as torch_cluster:

    radius(x, y, r, batch_x, batch_y)   -> [2, E]  row 0 indexes y, row 1 indexes x
    radius_graph(x, r, batch)           -> [2, E]  row 0 = neighbour (source),
                                                   row 1 = centre (target)

Neighbour cap. torch_cluster keeps at most `max_num_neighbors` (default 32) per
query point and silently drops the rest. QM9 molecules have at most 29 atoms, so on
QM9 the cap can never bind and both versions return the same edge set. On larger
molecules (GEOM-Drugs) it can bind. This version keeps every neighbour and warns,
so a difference shows up instead of passing silently.

Batches. PyG batches are contiguous (all atoms of molecule 0, then molecule 1, ...).
Distances are computed one molecule at a time, so memory grows with molecule size,
not with batch size.

Run this file directly for the self-test.
"""

import warnings

import torch

_warned = False


def _ptr(batch, num_graphs):
    """Start/end offsets of each molecule in a contiguous batch vector."""
    if batch.numel() > 1 and not bool((batch[1:] >= batch[:-1]).all()):
        raise ValueError("batch vector must be sorted (contiguous molecules)")
    counts = torch.bincount(batch, minlength=num_graphs)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).tolist()


def _warn_cap(counts, cap):
    global _warned
    if not _warned and counts.numel() and int(counts.max()) > cap:
        warnings.warn(f"a point has {int(counts.max())} neighbours, above torch_cluster's "
                      f"cap of {cap}. torch_cluster would drop some; this keeps all of them.")
        _warned = True


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32, num_workers=1):
    """For every point in y, all points in x within distance r, in the same molecule."""
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)
    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)
    if x.size(0) == 0 or y.size(0) == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    num_graphs = int(max(batch_x.max(), batch_y.max())) + 1
    px, py = _ptr(batch_x, num_graphs), _ptr(batch_y, num_graphs)

    rows, cols = [], []
    for g in range(num_graphs):
        xs, xe, ys, ye = px[g], px[g + 1], py[g], py[g + 1]
        if xe == xs or ye == ys:
            continue
        hit = torch.cdist(y[ys:ye], x[xs:xe]) < r          # [n_y, n_x]
        _warn_cap(hit.sum(1), max_num_neighbors)
        iy, ix = hit.nonzero(as_tuple=True)
        rows.append(iy + ys)
        cols.append(ix + xs)

    if not rows:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)
    return torch.stack([torch.cat(rows), torch.cat(cols)]).long()


def radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32,
                 flow="source_to_target", num_workers=1):
    """All pairs of atoms within distance r, in the same molecule."""
    if flow not in ("source_to_target", "target_to_source"):
        raise ValueError(flow)
    centre, neighbour = radius(x, x, r, batch, batch,
                               max_num_neighbors + (0 if loop else 1))
    if not loop:
        keep = centre != neighbour
        centre, neighbour = centre[keep], neighbour[keep]
    if flow == "source_to_target":
        return torch.stack([neighbour, centre])
    return torch.stack([centre, neighbour])


# ------------------------------------------------------------------
#  self-test
# ------------------------------------------------------------------

def _brute_radius(x, y, r, bx, by):
    out = set()
    for i in range(len(y)):
        for j in range(len(x)):
            if bx[j] == by[i] and float((y[i] - x[j]).norm()) < r:
                out.add((i, j))
    return out


def _edges(ei):
    return {tuple(e) for e in ei.t().tolist()}


def _self_test():
    torch.manual_seed(0)
    sizes = [12, 29, 0, 7, 1]            # includes an empty and a single-atom molecule
    x = torch.randn(sum(sizes), 3) * 1.6
    bx = torch.cat([torch.full((n,), g) for g, n in enumerate(sizes)]).long()
    y = torch.randn(15, 3) * 1.6          # e.g. bond centres
    by = torch.tensor([0] * 5 + [1] * 8 + [3] * 2).long()   # molecule 2 and 4 have none
    r = 2.5

    # radius vs brute force
    got = _edges(radius(x, y, r, bx, by))
    assert got == _brute_radius(x, y, r, bx, by), "radius != brute force"

    # radius_graph vs brute force (no self loops, both directions present)
    rg = radius_graph(x, r, bx)
    want = {(j, i) for (i, j) in _brute_radius(x, x, r, bx, bx) if i != j}
    assert _edges(rg) == want, "radius_graph != brute force"

    # against the real torch_cluster, if it happens to be installed
    try:
        import torch_cluster
        assert _edges(torch_cluster.radius(x, y, r, bx, by)) == got
        assert _edges(torch_cluster.radius_graph(x, r, bx)) == _edges(rg)
        ref = "identical to torch_cluster and to brute force"
    except ImportError:
        ref = "identical to brute force (torch_cluster not installed, which is expected)"

    if torch.cuda.is_available():
        gx, gy, gbx, gby = x.cuda(), y.cuda(), bx.cuda(), by.cuda()
        assert _edges(radius(gx, gy, r, gbx, gby).cpu()) == got, "GPU radius differs"
        assert _edges(radius_graph(gx, r, gbx).cpu()) == _edges(rg), "GPU radius_graph differs"
        ref += "; GPU matches CPU"

    print(f"torch_radius self-test PASS: {len(got)} radius edges, "
          f"{rg.shape[1]} radius_graph edges, {ref}")


if __name__ == "__main__":
    _self_test()
