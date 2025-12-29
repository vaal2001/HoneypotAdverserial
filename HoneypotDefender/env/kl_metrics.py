import numpy as np

from .host_profiles import HostType

def _safe_mean_std(values, eps = 1e-3):
    if len(values) == 0:
        return 0.0, eps
    arr = np.asarray(values, dtype=float)
    mu = float(np.mean(arr))
    std = float(np.std(arr))
    return mu, max(std, eps)

def _kl_gaussian(mu_p, std_p, mu_q, std_q):
    var_p = std_p ** 2
    var_q = std_q ** 2
    var_p = max(var_p, 1e-9)
    var_q = max(var_q, 1e-9)

    return 0.5 * ((var_p / var_q) + ((mu_q - mu_p) ** 2) / var_q - 1.0 + np.log(var_q / var_p))

def kl_real_vs_honeypot(hosts, adjacency, N_actual):
    """
    KL-divergence between REAL vs HONEYPOT host distributions
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

    if len(real["rtt_std"]) == 0 or len(honey["rtt_std"]) == 0:
        return 0.0

    features = ["rtt_mean", "rtt_std", "banner", "os", "art", "deg"]
    kls = []

    for f in features:
        mu_r, std_r = _safe_mean_std(real[f])
        mu_h, std_h = _safe_mean_std(honey[f])
        kls.append(_kl_gaussian(mu_r, std_r, mu_h, std_h))

    kl_val = float(np.mean(kls))
    return float(np.clip(kl_val, 0.0, 20.0))
