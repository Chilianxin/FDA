from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def moralize_graph(directed_adj: np.ndarray) -> np.ndarray:
    """
    Moralize a directed graph: connect co-parents and drop directions.

    directed_adj: [N, N] where entry (i, j) is strength from i -> j

    Returns undirected adjacency [N, N] with symmetric entries.
    Edge strengths are combined by max when multiple sources imply the same undirected edge.
    """
    if directed_adj.ndim != 2 or directed_adj.shape[0] != directed_adj.shape[1]:
        raise ValueError("directed_adj must be square [N, N]")
    num_nodes = directed_adj.shape[0]

    # Step 1: connect co-parents for each child
    undirected = np.zeros_like(directed_adj, dtype=np.float32)

    for child in range(num_nodes):
        parents = np.where(directed_adj[:, child] > 0)[0]
        # fully connect among parents (clique)
        for i_idx in range(len(parents)):
            for j_idx in range(i_idx + 1, len(parents)):
                a = parents[i_idx]
                b = parents[j_idx]
                strength = max(directed_adj[a, child], directed_adj[b, child])
                if strength > undirected[a, b]:
                    undirected[a, b] = strength
                    undirected[b, a] = strength

    # Step 2: drop directions by symmetrizing original edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            strength = max(directed_adj[i, j], directed_adj[j, i])
            if strength > undirected[i, j]:
                undirected[i, j] = strength
                undirected[j, i] = strength

    # Zero out diagonal
    np.fill_diagonal(undirected, 0.0)
    return undirected


def to_edge_index(undirected_adj: np.ndarray, threshold: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert undirected adjacency to torch_geometric edge_index and edge_weight tensors.

    threshold: only keep edges with weight > threshold.
    Returns (edge_index [2, E], edge_weight [E]). Duplicates (i, j) and (j, i) are both included
    to represent an undirected graph in PyG.
    """
    if undirected_adj.ndim != 2 or undirected_adj.shape[0] != undirected_adj.shape[1]:
        raise ValueError("undirected_adj must be square [N, N]")
    num_nodes = undirected_adj.shape[0]

    sources = []
    targets = []
    weights = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            w = float(undirected_adj[i, j])
            if w > threshold:
                sources.append(i)
                targets.append(j)
                weights.append(w)

    if len(sources) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_weight = torch.empty((0,), dtype=torch.float32)
    else:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight

