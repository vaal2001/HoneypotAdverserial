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
    Generates a connected random network of size N_actual <= N_max.

    Returns:
        - N_actual: number of real nodes in this episode
        - hosts: list of Host objects of length N_actual
        - adjacency_padded: (N_max, N_max) matrix, adjacency padded with zeros
    """
    honeypot_ratio = rng.uniform(0.0, 1.0)

    # ------------------------------------------
    # Determine number of nodes
    # ------------------------------------------
    if max_nodes is None:
        max_nodes = N_max

    N_actual = int(rng.integers(min_nodes, max_nodes + 1))
    N_actual = max(1, min(N_actual, N_max))  # safety clamp

    # ------------------------------------------
    # Choose honeypot positions
    # ------------------------------------------
    num_honeypots = int(round(honeypot_ratio * N_actual))
    num_honeypots = max(0, min(num_honeypots, N_actual))

    honeypot_indices = set(
        rng.choice(N_actual, size=num_honeypots, replace=False)
    )

    # ------------------------------------------
    # Create host objects
    # ------------------------------------------
    hosts: List[Host] = []
    for i in range(N_actual):
        is_honey = i in honeypot_indices
        hosts.append(generate_random_host(i, is_honey, rng))

    # ------------------------------------------
    # Create connected random graph
    # ------------------------------------------
    # Erdős–Rényi initial graph
    G = nx.erdos_renyi_graph(N_actual, p=0.3)

    # Ensure connectivity
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        # Chain connect components (minimum edges to make graph connected)
        for c1, c2 in zip(components[:-1], components[1:]):
            G.add_edge(next(iter(c1)), next(iter(c2)))

    # Convert to adjacency matrix
    adj = nx.to_numpy_array(G, dtype=int)

    # ------------------------------------------
    # Pad adjacency to N_max × N_max
    # ------------------------------------------
    adjacency_padded = np.zeros((N_max, N_max), dtype=int)
    adjacency_padded[:N_actual, :N_actual] = adj

    return N_actual, hosts, adjacency_padded
