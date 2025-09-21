from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List

try:
    import igraph as ig
    import leidenalg as la
except Exception:
    ig = None
    la = None
try:
    import networkx as nx
    import community as nx_comm
except Exception:
    nx = None
    nx_comm = None


def detect_communities(S: np.ndarray, resolution: float = 1.0) -> np.ndarray:
    if ig is not None and la is not None:
        g = ig.Graph.Weighted_Adjacency(S.tolist(), mode=ig.ADJ_UNDIRECTED, attr='weight', loops=False)
        part = la.find_partition(g, la.RBConfigurationVertexPartition, weights='weight', resolution_parameter=resolution)
        return np.array(part.membership)
    if nx is not None and nx_comm is not None:
        G = nx.from_numpy_array(S)
        part = nx_comm.best_partition(G, weight='weight', resolution=resolution)
        return np.array([part[i] for i in range(len(part))])
    raise ImportError('No community backend available')


def align_labels(prev: np.ndarray | None, curr: np.ndarray) -> np.ndarray:
    if prev is None:
        return curr
    prev_ids = np.unique(prev)
    curr_ids = np.unique(curr)
    overlap = np.zeros((len(prev_ids), len(curr_ids)), dtype=int)
    for i, pid in enumerate(prev_ids):
        for j, cid in enumerate(curr_ids):
            overlap[i, j] = np.sum((prev == pid) & (curr == cid))
    used_i, used_j = set(), set()
    mapping: Dict[int, int] = {}
    while True:
        i, j = divmod(overlap.argmax(), overlap.shape[1])
        if overlap[i, j] <= 0:
            break
        if i in used_i or j in used_j:
            overlap[i, j] = -1
            continue
        mapping[curr_ids[j]] = int(prev_ids[i])
        used_i.add(i); used_j.add(j)
        overlap[i, :] = -1; overlap[:, j] = -1
    out = curr.copy()
    next_id = int(prev_ids.max()) + 1 if len(prev_ids) else 0
    for cid in curr_ids:
        if cid in mapping:
            out[curr == cid] = mapping[cid]
        else:
            out[curr == cid] = next_id; next_id += 1
    return out


def smooth_labels(history: List[np.ndarray], window: int = 4) -> np.ndarray:
    H = history[-window:]
    if len(H) == 1:
        return H[0]
    base = H[0]
    aligned = [base]
    for arr in H[1:]:
        aligned.append(align_labels(aligned[-1], arr))
    A = np.stack(aligned, axis=1)
    out = np.zeros(A.shape[0], dtype=int)
    for i in range(A.shape[0]):
        vals, cnts = np.unique(A[i], return_counts=True)
        out[i] = int(vals[np.argmax(cnts)])
    return out


def merge_small(labels: np.ndarray, min_size: int = 5) -> np.ndarray:
    uniq, cnt = np.unique(labels, return_counts=True)
    small = [int(u) for u, c in zip(uniq, cnt) if c < min_size]
    if not small:
        return labels
    large = [int(u) for u, c in zip(uniq, cnt) if c >= min_size]
    if not large:
        return labels
    tgt = large[0]
    out = labels.copy()
    for s in small:
        out[out == s] = tgt
    return out

