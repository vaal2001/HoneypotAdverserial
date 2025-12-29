import numpy as np
import networkx as nx

from .host_profiles import generate_random_host

def generate_random_network(N_max, rng, min_nodes = 5, max_nodes = None, ):
    """
    Generates a random network of hosts.
    """
    if max_nodes is None:
        max_nodes = N_max

    N_actual = int(rng.integers(min_nodes, max_nodes + 1))
    N_actual = max(1, min(N_actual, N_max))

    honeypot_ratio = float(rng.uniform(0.1, 0.6))
    num_honeypots = int(round(honeypot_ratio * N_actual))
    num_honeypots = max(1, min(num_honeypots, N_actual))

    honeypot_indices = set(rng.choice(N_actual, size=num_honeypots, replace=False))

    hosts = []
    for i in range(N_actual):
        is_h = i in honeypot_indices
        hosts.append(generate_random_host(i, is_h, rng))

    G = nx.erdos_renyi_graph(N_actual, p=0.3)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for c1, c2 in zip(comps[:-1], comps[1:]):
            G.add_edge(next(iter(c1)), next(iter(c2)))

    adj = nx.to_numpy_array(G, dtype=int)

    adjacency_padded = np.zeros((N_max, N_max), dtype=int)
    adjacency_padded[:N_actual, :N_actual] = adj

    return N_actual, hosts, adjacency_padded
