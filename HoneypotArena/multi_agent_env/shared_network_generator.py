# shared_network_generator.py
import numpy as np
import networkx as nx
from typing import List, Tuple
from .shared_host_profiles import Host, HostType, ResponseModel

DEVICE_TYPES = ["linux_web", "win_desktop", "iot_sensor", "db_server"]

def _rand_service_count(device, rng):
    if device == "linux_web": return int(rng.integers(1, 4))
    if device == "win_desktop": return int(rng.integers(1, 3))
    if device == "iot_sensor": return 1
    if device == "db_server": return int(rng.integers(1, 4))
    return int(rng.integers(1, 3))

def generate_random_host(host_id, is_honey, rng):
    device = rng.choice(DEVICE_TYPES)

    if is_honey:
        rm = ResponseModel(
            rtt_mean=float(rng.uniform(10,80)),
            rtt_std=float(rng.uniform(1,8)),
            banner_noise=float(rng.uniform(0,0.2)),
            os_fingerprint=float(rng.normal(0.7,0.1)),
            artefact_prob=float(rng.uniform(0.1,0.4)),
            service_count=_rand_service_count(device, rng)
        )
        ht = HostType.HONEYPOT
    else:
        rm = ResponseModel(
            rtt_mean=float(rng.uniform(10,80)),
            rtt_std=float(rng.uniform(5,20)),
            banner_noise=float(rng.uniform(0.05,0.25)),
            os_fingerprint=float(rng.normal(1.0,0.1)),
            artefact_prob=float(rng.uniform(0,0.02)),
            service_count=_rand_service_count(device, rng)
        )
        ht = HostType.REAL

    return Host(host_id, ht, device, rm)

def generate_random_network(N_max, rng, min_nodes=5, max_nodes=None):
    if max_nodes is None: max_nodes = N_max
    N_actual = int(rng.integers(min_nodes, max_nodes+1))

    honey_ratio = float(rng.uniform(0.1,0.6))
    num_honey = int(round(honey_ratio * N_actual))
    num_honey = max(1, min(num_honey, N_actual))

    honey_idx = set(rng.choice(N_actual, size=num_honey, replace=False))

    hosts = [generate_random_host(i, i in honey_idx, rng) for i in range(N_actual)]

    G = nx.erdos_renyi_graph(N_actual, p=0.3)
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for c1, c2 in zip(comps[:-1], comps[1:]):
            G.add_edge(next(iter(c1)), next(iter(c2)))

    adj = nx.to_numpy_array(G, dtype=int)

    pad = np.zeros((N_max, N_max), dtype=int)
    pad[:N_actual, :N_actual] = adj

    return N_actual, hosts, pad
