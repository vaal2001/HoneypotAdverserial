from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .host_profiles import Host, HostType


@dataclass
class RealBaselineStats:
    """
    Baseline statistieken van REAL hosts in een episode.
    Wordt gebruikt om 'distance from real' te meten.
    """
    rtt_std_mean: float
    rtt_std_std: float
    banner_noise_mean: float
    banner_noise_std: float
    os_fp_mean: float
    os_fp_std: float
    artefact_prob_mean: float
    artefact_prob_std: float
    service_count_mean: float
    service_count_std: float
    degree_mean: float
    degree_std: float


def _safe_std(x: np.ndarray, eps: float = 1e-3) -> float:
    if x.size == 0:
        return eps
    s = float(np.std(x))
    return max(s, eps)


def compute_real_baseline_stats(
    hosts: List[Host],
    adjacency: np.ndarray,
    N_actual: int,
) -> RealBaselineStats:
    """
    Bepaal per episode de typische distributie van REAL hosts.
    """
    # degree per node
    deg = adjacency[:N_actual, :N_actual].sum(axis=1).astype(float)

    rtt_std_list = []
    banner_noise_list = []
    os_fp_list = []
    artefact_list = []
    svc_list = []
    deg_list = []

    for i, h in enumerate(hosts[:N_actual]):
        if h.host_type != HostType.REAL:
            continue
        rm = h.response_model
        rtt_std_list.append(rm.rtt_std)
        banner_noise_list.append(rm.banner_noise)
        os_fp_list.append(rm.os_fingerprint)
        artefact_list.append(rm.artefact_prob)
        svc_list.append(rm.service_count)
        deg_list.append(deg[i])

    # Fallback als er (bijna) geen real hosts zijn
    if len(rtt_std_list) == 0:
        rtt_std_list = [8.0]
        banner_noise_list = [0.12]
        os_fp_list = [1.0]
        artefact_list = [0.01]
        svc_list = [2.0]
        deg_list = [2.0]

    rtt_std_arr = np.array(rtt_std_list, dtype=float)
    banner_arr = np.array(banner_noise_list, dtype=float)
    os_arr = np.array(os_fp_list, dtype=float)
    artefact_arr = np.array(artefact_list, dtype=float)
    svc_arr = np.array(svc_list, dtype=float)
    deg_arr = np.array(deg_list, dtype=float)

    return RealBaselineStats(
        rtt_std_mean=float(np.mean(rtt_std_arr)),
        rtt_std_std=_safe_std(rtt_std_arr),
        banner_noise_mean=float(np.mean(banner_arr)),
        banner_noise_std=_safe_std(banner_arr),
        os_fp_mean=float(np.mean(os_arr)),
        os_fp_std=_safe_std(os_arr),
        artefact_prob_mean=float(np.mean(artefact_arr)),
        artefact_prob_std=_safe_std(artefact_arr),
        service_count_mean=float(np.mean(svc_arr)),
        service_count_std=_safe_std(svc_arr),
        degree_mean=float(np.mean(deg_arr)),
        degree_std=_safe_std(deg_arr),
    )


def _clip_z(z: float, z_max: float = 4.0) -> float:
    return float(min(max(z, 0.0), z_max))


def detectability_single(
    host: Host,
    degree: float,
    baseline: RealBaselineStats,
) -> float:
    """
    Heuristische detectability-score (0..~10):
      hogere waarde = meer afwijkend van 'typische' REAL host.

    Dit is bedoeld als conventionele attacker-heuristiek.
    """
    rm = host.response_model

    # z-scores t.o.v. real-baseline
    z_rtt_std = _clip_z(abs(rm.rtt_std - baseline.rtt_std_mean) / baseline.rtt_std_std)
    z_banner = _clip_z(abs(rm.banner_noise - baseline.banner_noise_mean) / baseline.banner_noise_std)
    z_os = _clip_z(abs(rm.os_fingerprint - baseline.os_fp_mean) / baseline.os_fp_std)
    z_svc = _clip_z(abs(rm.service_count - baseline.service_count_mean) / baseline.service_count_std)
    z_deg = _clip_z(abs(degree - baseline.degree_mean) / baseline.degree_std)

    # extra penalties
    # Honeypots hebben vaak té lage jitter (uniform RTT)
    low_jitter = max(0.0, (baseline.rtt_std_mean - rm.rtt_std) / (baseline.rtt_std_mean + 1e-3))
    low_jitter_penalty = min(low_jitter, 2.0)  # cap

    # Artefact penalty (zwaar wegen, maar gecapt)
    artefact_penalty = min(rm.artefact_prob * 5.0, 3.0)

    # combineer z-scores met gewichten
    score = (
        0.8 * z_rtt_std +
        0.6 * z_banner +
        1.0 * z_os +
        0.4 * z_svc +
        0.4 * z_deg +
        1.0 * low_jitter_penalty +
        1.5 * artefact_penalty
    )

    # cap totale detectability
    score = float(min(max(score, 0.0), 10.0))
    return score


def detectability_for_hosts(
    hosts: List[Host],
    adjacency: np.ndarray,
    baseline: RealBaselineStats,
    N_actual: int,
) -> Tuple[float, float]:
    """
    Gemiddelde detectability:
      - honeypots
      - real hosts

    Waarden zitten grofweg in [0, 10].
    """
    deg = adjacency[:N_actual, :N_actual].sum(axis=1).astype(float)

    honey_scores = []
    real_scores = []

    for i, h in enumerate(hosts[:N_actual]):
        s = detectability_single(h, degree=deg[i], baseline=baseline)
        if h.host_type == HostType.HONEYPOT:
            honey_scores.append(s)
        else:
            real_scores.append(s)

    mean_honey = float(np.mean(honey_scores)) if honey_scores else 0.0
    mean_real = float(np.mean(real_scores)) if real_scores else 0.0
    return mean_honey, mean_real
