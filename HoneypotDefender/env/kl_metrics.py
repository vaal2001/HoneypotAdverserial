from __future__ import annotations
from typing import List
import numpy as np

from .host_profiles import Host, HostType


def _safe_mean_std(values, eps: float = 1e-3):
    if len(values) == 0:
        return 0.0, eps
    arr = np.asarray(values, dtype=float)
    mu = float(np.mean(arr))
    std = float(np.std(arr))
    return mu, max(std, eps)


def _kl_gaussian(mu_p: float, std_p: float, mu_q: float, std_q: float) -> float:
    """
    KL(N_p || N_q) (1D).
    """
    var_p = std_p ** 2
    var_q = std_q ** 2
    var_p = max(var_p, 1e-9)
    var_q = max(var_q, 1e-9)

    return 0.5 * (
        (var_p / var_q)
        + ((mu_q - mu_p) ** 2) / var_q
        - 1.0
        + np.log(var_q / var_p)
    )


def kl_real_vs_honeypot(
    hosts: List[Host],
    adjacency: np.ndarray,
    N_actual: int,
) -> float:
    """
    KL-divergence tussen distributies van REAL vs HONEYPOT hosts
    over enkele kernfeatures:
      - rtt_mean
      - rtt_std
      - banner_noise
      - os_fingerprint
      - artefact_prob
      - degree
    """
    deg = adjacency[:N_actual, :N_actual].sum(axis=1).astype(float)

    real = {"rtt_mean": [], "rtt_std": [], "banner": [], "os": [], "art": [], "deg": []}
    honey = {"rtt_mean": [], "rtt_std": [], "banner": [], "os": [], "art": [], "deg": []}

    for i, h in enumerate(hosts[:N_actual]):
        rm = h.response_model
        if h.host_type == HostType.REAL:
            real["rtt_mean"].append(rm.rtt_mean)
            real["rtt_std"].append(rm.rtt_std)
            real["banner"].append(rm.banner_noise)
            real["os"].append(rm.os_fingerprint)
            real["art"].append(rm.artefact_prob)
            real["deg"].append(deg[i])
        else:
            honey["rtt_mean"].append(rm.rtt_mean)
            honey["rtt_std"].append(rm.rtt_std)
            honey["banner"].append(rm.banner_noise)
            honey["os"].append(rm.os_fingerprint)
            honey["art"].append(rm.artefact_prob)
            honey["deg"].append(deg[i])

    # Als één van de groepen leeg is → KL = 0
    if len(real["rtt_std"]) == 0 or len(honey["rtt_std"]) == 0:
        return 0.0

    features = ["rtt_mean", "rtt_std", "banner", "os", "art", "deg"]
    kls = []

    for f in features:
        mu_r, std_r = _safe_mean_std(real[f])
        mu_h, std_h = _safe_mean_std(honey[f])
        kls.append(_kl_gaussian(mu_r, std_r, mu_h, std_h))

    kl_val = float(np.mean(kls))
    # clip om runaway KL te voorkomen
    return float(np.clip(kl_val, 0.0, 20.0))
