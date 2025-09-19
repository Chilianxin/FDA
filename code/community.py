from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

try:
    import igraph as ig
    import leidenalg as la
except Exception:
    ig = None
    la = None
try:
    import networkx as nx
    import community as nx_comm  # python-louvain
except Exception:
    nx = None
    nx_comm = None


def leiden_communities(S: np.ndarray, resolution: float = 1.0) -> np.ndarray:
    if ig is not None and la is not None:
        g = ig.Graph.Weighted_Adjacency(S.tolist(), mode=ig.ADJ_UNDIRECTED, attr="weight", loops=False)
        part = la.find_partition(g, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution)
        labels = np.array(part.membership)
        return labels
    # Fallback to Louvain via networkx if igraph/leidenalg not available
    if nx is not None and nx_comm is not None:
        G = nx.from_numpy_array(S)
        part = nx_comm.best_partition(G, weight='weight', resolution=resolution)
        # part is dict: node -> community id
        labels = np.array([part[i] for i in range(len(part))])
        return labels
    raise ImportError("No community detection backend available. Install igraph+leidenalg or python-louvain.")


def align_labels(prev_labels: np.ndarray, curr_labels: np.ndarray) -> np.ndarray:
    """Align curr_labels to prev_labels using maximum overlap (Hungarian-like by greedy).
    For robustness and simplicity, use a greedy overlap matching.
    """
    if prev_labels is None:
        return curr_labels
    prev = prev_labels
    curr = curr_labels
    prev_ids = np.unique(prev)
    curr_ids = np.unique(curr)
    # build overlap matrix
    overlap = np.zeros((len(prev_ids), len(curr_ids)), dtype=int)
    for i, pid in enumerate(prev_ids):
        for j, cid in enumerate(curr_ids):
            overlap[i, j] = np.sum((prev == pid) & (curr == cid))
    # greedy match
    used_prev, used_curr = set(), set()
    mapping: Dict[int, int] = {}
    while True:
        i, j = divmod(overlap.argmax(), overlap.shape[1])
        if overlap[i, j] == 0:
            break
        if i in used_prev or j in used_curr:
            overlap[i, j] = -1
            continue
        mapping[curr_ids[j]] = prev_ids[i]
        used_prev.add(i)
        used_curr.add(j)
        overlap[i, :] = -1
        overlap[:, j] = -1
    # assign mapped labels; new labels get new ids after max prev
    out = curr.copy()
    next_id = int(prev_ids.max()) + 1 if len(prev_ids) > 0 else 0
    for cid in curr_ids:
        if cid in mapping:
            out[curr == cid] = mapping[cid]
        else:
            out[curr == cid] = next_id
            next_id += 1
    return out


def smooth_membership(history: List[np.ndarray], window: int = 4) -> np.ndarray:
    """Majority voting over last 'window' snapshots. history[-1] is current proposal."""
    if len(history) == 0:
        raise ValueError("history cannot be empty")
    H = history[-window:]
    # Pad to same label set by remapping sequentially
    base = H[0]
    aligned = [base]
    for arr in H[1:]:
        aligned.append(align_labels(aligned[-1], arr))
    A = np.stack(aligned, axis=1)
    # majority vote per node
    out = np.zeros(A.shape[0], dtype=int)
    for i in range(A.shape[0]):
        vals, cnts = np.unique(A[i], return_counts=True)
        out[i] = int(vals[np.argmax(cnts)])
    return out


def merge_small_clusters(labels: np.ndarray, min_size: int = 5) -> np.ndarray:
    labels = labels.copy()
    unique, counts = np.unique(labels, return_counts=True)
    small = set(unique[counts < min_size])
    if not small:
        return labels
    # reassign small to nearest large by id (placeholder; in practice could use centroid distances)
    large = [int(u) for u, c in zip(unique, counts) if c >= min_size]
    if not large:
        return labels
    target = large[0]
    for s in small:
        labels[labels == s] = target
    return labels

