"""
check_env.py - Step 1 pass/fail check for the RTX 5080 machine (native Windows).

Run with the project venv:   ..\.venv\Scripts\python.exe check_env.py

Every check imports a library AND runs a small operation with it on the GPU.
Importing is not enough: a torch build without Blackwell (sm_120) kernels imports
fine and only fails the first time it launches a kernel.

torch_scatter and torch_cluster are deliberately NOT required. Torsional Diffusion
uses them in three places; we replace them with plain PyTorch in step 2. The two
checks below for scatter and neighbour search test those replacements.

Exit code 0 = everything passed.
"""

import platform
import sys
import traceback


def square(x):
    # top level so it can be pickled by Windows' "spawn" multiprocessing
    return x * x


# ------------------------------------------------------------------

def python_version():
    v = sys.version_info
    if (v.major, v.minor) < (3, 10):
        raise RuntimeError(f"Python {v.major}.{v.minor}; need 3.10+ (3.11 recommended)")
    return f"{v.major}.{v.minor}.{v.micro} on {platform.platform()}"


def torch_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"torch {torch.__version__} cannot see a GPU "
                           f"(built for CUDA {torch.version.cuda})")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    cuda = torch.version.cuda
    major, minor = (int(x) for x in cuda.split(".")[:2])
    if (major, minor) < (12, 8):
        raise RuntimeError(f"torch built for CUDA {cuda}; the 5080 needs 12.8+")
    arch = f"sm_{cap[0]}{cap[1]}"
    if arch not in torch.cuda.get_arch_list():
        raise RuntimeError(f"GPU is {arch} but this torch build only has kernels for "
                           f"{torch.cuda.get_arch_list()}")
    return f"torch {torch.__version__}, CUDA {cuda}, {name} ({arch})"


def gpu_math():
    import torch
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    gpu = (a.cuda() @ b.cuda()).cpu()
    err = (gpu - a @ b).abs().max().item()
    if err > 1e-2:
        raise RuntimeError(f"GPU matmul disagrees with CPU by {err}")
    x = torch.randn(64, 32, device="cuda", requires_grad=True)
    (x.sin() ** 2).sum().backward()
    if x.grad is None:
        raise RuntimeError("backward produced no gradient")
    return f"matmul max|gpu-cpu| = {err:.2e}, backward ok"


def pyg():
    import torch
    import torch_geometric
    from torch_geometric.data import Data, Batch
    d = Data(x=torch.randn(5, 4), edge_index=torch.tensor([[0, 1], [1, 2]]))
    b = Batch.from_data_list([d, d]).to("cuda")
    return f"torch_geometric {torch_geometric.__version__}, batched {b.num_nodes} nodes on GPU"


def pyg_scatter():
    """The replacement for torch_scatter.scatter, same call signature."""
    import importlib.util
    import torch
    from torch_geometric.utils import scatter
    src = torch.ones(10, device="cuda")
    idx = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3], device="cuda")
    out = scatter(src, idx, dim=0, dim_size=5, reduce="sum")
    if out.tolist() != [2.0, 3.0, 4.0, 1.0, 0.0]:
        raise RuntimeError(f"wrong scatter result {out.tolist()}")
    mean = scatter(src * 2, idx, dim=0, dim_size=5, reduce="mean")
    if mean[:4].tolist() != [2.0, 2.0, 2.0, 2.0]:
        raise RuntimeError(f"wrong scatter-mean result {mean.tolist()}")
    has_ts = importlib.util.find_spec("torch_scatter") is not None
    note = " (torch_scatter is installed but not needed)" if has_ts else ""
    return "sum and mean correct on GPU, no torch_scatter needed" + note


def neighbour_search():
    """Preview of the torch_cluster.radius_graph replacement: cdist + same-molecule mask.

    Checked against a slow CPU double loop on a fake batch of 3 molecules.
    """
    import torch
    torch.manual_seed(0)
    sizes = [12, 29, 7]                      # QM9 molecules have at most 29 atoms
    pos = torch.randn(sum(sizes), 3) * 1.5
    batch = torch.cat([torch.full((n,), i) for i, n in enumerate(sizes)])
    r = 2.0

    p, bt = pos.cuda(), batch.cuda()
    d = torch.cdist(p, p)
    mask = (d < r) & (bt[:, None] == bt[None, :])
    mask.fill_diagonal_(False)
    gpu_edges = {tuple(e) for e in mask.nonzero().cpu().tolist()}

    cpu_edges = set()
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i != j and batch[i] == batch[j] and (pos[i] - pos[j]).norm() < r:
                cpu_edges.add((i, j))

    if gpu_edges != cpu_edges:
        raise RuntimeError(f"GPU found {len(gpu_edges)} edges, brute force {len(cpu_edges)}")
    return f"{len(gpu_edges)} edges, identical to brute force"


def e3nn_check():
    import torch
    import e3nn
    from e3nn import o3
    tp = o3.FullyConnectedTensorProduct("1x1o", "1x1o", "1x0e").to("cuda")
    x = torch.randn(8, 3, device="cuda")
    y = tp(x, x)
    # a 1o x 1o -> 0e product is rotation invariant: rotating x must not change it
    R = o3.rand_matrix().to("cuda")
    y_rot = tp(x @ R.T, x @ R.T)
    err = (y - y_rot).abs().max().item()
    if err > 1e-4:
        raise RuntimeError(f"invariance broken, err={err}")
    return f"e3nn {e3nn.__version__}, rotation-invariance err {err:.1e}"


def rdkit_check():
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdMolTransforms
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))      # butane: one rotatable C-C bond
    if AllChem.EmbedMolecule(mol, randomSeed=0) != 0:
        raise RuntimeError("ETKDG embedding failed")
    conf = mol.GetConformer()
    before = rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3)
    rdMolTransforms.SetDihedralDeg(conf, 0, 1, 2, 3, 60.0)  # the operation we will use
    after = rdMolTransforms.GetDihedralDeg(conf, 0, 1, 2, 3)
    if abs(after - 60.0) > 1e-3:
        raise RuntimeError(f"set dihedral to 60, read back {after}")
    return f"rdkit {rdBase.rdkitVersion}, butane dihedral {before:.1f} -> {after:.1f} deg"


def spyrmsd_check():
    import spyrmsd
    return f"spyrmsd {getattr(spyrmsd, '__version__', '?')}"


def multiprocessing_check():
    """Windows starts worker processes by 'spawn'. Evaluation and preprocessing use Pool."""
    import multiprocessing as mp
    with mp.Pool(2) as p:
        out = p.map(square, range(8))
    if out != [x * x for x in range(8)]:
        raise RuntimeError(f"pool returned {out}")
    return f"Pool(2) ok, start method '{mp.get_start_method()}'"


CHECKS = [
    ("Python 3.10+", python_version),
    ("PyTorch sees the 5080 (CUDA 12.8+, sm_120)", torch_cuda),
    ("GPU math + autograd", gpu_math),
    ("torch_geometric", pyg),
    ("scatter via torch_geometric (replaces torch_scatter)", pyg_scatter),
    ("neighbour search in plain PyTorch (replaces torch_cluster)", neighbour_search),
    ("e3nn on GPU", e3nn_check),
    ("RDKit embed + set torsion", rdkit_check),
    ("spyrmsd (evaluation)", spyrmsd_check),
    ("multiprocessing (Windows spawn)", multiprocessing_check),
]


def main():
    results = []
    for name, fn in CHECKS:
        try:
            results.append((name, True, fn() or ""))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()

    print("\n" + "=" * 78)
    print("  STEP 1 ENVIRONMENT CHECK")
    print("=" * 78)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if detail:
            print(f"         {detail}")
    print("=" * 78)
    n_fail = sum(not ok for _, ok, _ in results)
    print("  ALL PASS - step 1 done." if n_fail == 0 else
          f"  {n_fail} FAILED - send this whole output back before moving on.")
    return 1 if n_fail else 0


if __name__ == "__main__":      # required on Windows, or Pool workers re-run this file
    sys.exit(main())
