# Vendored from DeFoG (github.com/manuelmlmadeira/DeFoG, src/models/extra_features.py; MIT licence),
# which builds on DiGress. Changes: ExtraFeatures takes a binary adjacency A (bs, n, n) and a node mask
# instead of DeFoG's noisy_data dict, and returns (X, E, y) tuples; the pdb fallback is removed.
# Feature definitions and normalisations are unchanged.

import torch


class ExtraFeatures:
    """
    'rrwp'  (DeFoG default): RRWP edge features (k steps) + their diagonal as node features;
             graph features = [n / n_max, 3-6 cycle counts]
    'cycles': node 3-5 cycle counts; graph features = [n / n_max, 3-6 cycle counts]
    'all'   (DiGress default): cycles + Laplacian eigen-features
    'rrwp_comp': RRWP on the graph and on its complement (k/2 steps each)
    """

    def __init__(self, features_type, rrwp_steps, max_n_nodes):
        self.max_n_nodes = max_n_nodes
        self.features_type = features_type
        self.rrwp_steps = rrwp_steps
        self.ncycles = NodeCycleFeatures()
        self.RRWP = RRWPFeatures()
        if features_type == "all":
            self.eigenfeatures = EigenFeatures(mode="all")

    def dims(self):
        """(dX, dE, dy) of the extra features."""
        k = self.rrwp_steps
        return {"rrwp": (k, k, 5), "cycles": (3, 0, 5), "all": (6, 0, 11),
                "rrwp_comp": (2 * (k // 2), 2 * (k // 2), 5)}[self.features_type]

    def __call__(self, A, node_mask):
        A = A.float()
        n = node_mask.sum(dim=1).unsqueeze(1).float() / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(A, node_mask)
        empty_e = A.new_zeros((*A.shape, 0))

        if self.features_type == "cycles":
            return x_cycles, empty_e, torch.hstack((n, y_cycles))

        if self.features_type == "rrwp":
            rrwp_edge_attr = self.RRWP(A, k=self.rrwp_steps)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]
            return rrwp_node_attr, rrwp_edge_attr, torch.hstack((n, y_cycles))

        if self.features_type == "rrwp_comp":
            k = int(self.rrwp_steps / 2)
            rrwp_edge_attr = self.RRWP(A, k=k)
            diag_index = torch.arange(rrwp_edge_attr.shape[1])
            rrwp_node_attr = rrwp_edge_attr[:, diag_index, diag_index, :]
            comp_rrwp_edge_attr = self.RRWP(1 - A, k=k)
            comp_rrwp_node_attr = comp_rrwp_edge_attr[:, diag_index, diag_index, :]
            return (torch.cat((rrwp_node_attr, comp_rrwp_node_attr), dim=-1),
                    torch.cat((rrwp_edge_attr, comp_rrwp_edge_attr), dim=-1),
                    torch.hstack((n, y_cycles)))

        if self.features_type == "all":
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = self.eigenfeatures(A, node_mask)
            return (torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1), empty_e,
                    torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))

        raise ValueError(f"Features type {self.features_type} not implemented")


class RRWPFeatures:
    def __init__(self, k=10, normalize=True):
        self.k = k
        self.normalize = normalize

    def __call__(self, E, k=None):
        k = k or self.k
        bs, n, _ = E.shape
        if self.normalize:
            degree = torch.zeros(bs, n, n, device=E.device)
            to_fill = 1 / (E.sum(dim=-1).float())
            to_fill[E.sum(dim=-1).float() == 0] = 0
            degree = torch.diagonal_scatter(degree, to_fill, dim1=1, dim2=2)
            E = degree @ E

        id = torch.eye(n, device=E.device).unsqueeze(0).repeat(bs, 1, 1)
        rrwp_list = [id]
        for _ in range(k - 1):
            rrwp_list.append(rrwp_list[-1] @ E)
        return torch.stack(rrwp_list, -1)


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, adj_matrix, node_mask):
        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)
        x_cycles = x_cycles.type_as(adj_matrix) * node_mask.unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py"""

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, A, mask):
        A = A * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        eigvals, eigvectors = torch.linalg.eigh(L)
        eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
        eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
        n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
        nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(
            vectors=eigvectors, node_mask=mask, n_connected=n_connected_comp)
        return n_connected_comp.type_as(A), batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector


def compute_laplacian(adjacency, normalize: bool):
    diag = torch.sum(adjacency, dim=-1)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)
    combinatorial = D - adjacency
    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2
    diag0 = diag.clone()
    diag[diag == 0] = 1e-12
    diag_norm = 1 / torch.sqrt(diag)
    D_norm = torch.diag_embed(diag_norm)
    L = torch.eye(n, device=adjacency.device).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    bs, n = vectors.size(0), vectors.size(1)
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values
    mask = ~(first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)
    indices = indices.expand(-1, n, -1)
    first_k_ev = torch.gather(vectors, dim=2, index=indices)
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)
    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    return torch.diagonal(X, dim1=-2, dim2=-1).sum(dim=-1)


def batch_diagonal(X):
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """Builds cycle counts for each node in a graph."""

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = (diag_a5 - 2 * triangles * self.d
              - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles)
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t
                + 4 * term_7_t + 3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()
        k3x, k3y = self.k3_cycle()
        k4x, k4y = self.k4_cycle()
        k5x, k5y = self.k5_cycle()
        _, k6y = self.k6_cycle()
        return torch.cat([k3x, k4x, k5x], dim=-1), torch.cat([k3y, k4y, k5y, k6y], dim=-1)
