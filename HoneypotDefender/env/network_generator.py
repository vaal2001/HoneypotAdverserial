from __future__ import annotations
from typing import List, Tuple
import numpy as np
import networkx as nx

from .host_profiles import Host, generate_random_host


def generate_random_network(
    N_max: int,
    rng: np.random.Generator,
    min_nodes: int = 5,
    max_nodes: int | None = None,
) -> Tuple[int, List[Host], np.ndarray]:
    """
    Genereert een verbonden random netwerk met N_actual <= N_max.

    Returns:
        - N_actual: aantal echte nodes
        - hosts: lijst Host objects (len = N_actual)
        - adjacency_padded: (N_max, N_max) adjacency matrix met zero-padding
    """
    if max_nodes is None:
        max_nodes = N_max

    N_actual = int(rng.integers(min_nodes, max_nodes + 1))
    N_actual = max(1, min(N_actual, N_max))

    # kies honeypot ratio
    honeypot_ratio = float(rng.uniform(0.1, 0.6))
    num_honeypots = int(round(honeypot_ratio * N_actual))
    num_honeypots = max(1, min(num_honeypots, N_actual))

    honeypot_indices = set(rng.choice(N_actual, size=num_honeypots, replace=False))

    hosts: List[Host] = []
    for i in range(N_actual):
        is_h = i in honeypot_indices
        hosts.append(generate_random_host(i, is_h, rng))

    # random connected graph
    G = nx.erdos_renyi_graph(N_actual, p=0.3)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for c1, c2 in zip(comps[:-1], comps[1:]):
            G.add_edge(next(iter(c1)), next(iter(c2)))

    adj = nx.to_numpy_array(G, dtype=int)

    adjacency_padded = np.zeros((N_max, N_max), dtype=int)
    adjacency_padded[:N_actual, :N_actual] = adj

    return N_actual, hosts, adjacency_padded
