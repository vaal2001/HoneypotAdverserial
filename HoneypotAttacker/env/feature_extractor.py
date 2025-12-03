import numpy as np
from .probe_models import ProbeResult

def init_feature_matrix(N_max: int, F: int) -> np.ndarray:
    return np.zeros((N_max, F), dtype=np.float32)


def update_features(
    features: np.ndarray,
    host_idx: int,
    probe_type: int,
    result: ProbeResult,
) -> None:
    """
    Verbeterde features:

      f[0] : total probe count
      f[1] : open_count
      f[2] : closed_count
      f[3] : rtt_mean
      f[4] : rtt_var_accum   (for Welford)
      f[5] : max banner score
      f[6] : max os score
      f[7] : honeypot artefact counter
      f[8] : probe diversity (distinct probes)
      f[9] : |banner - os| inconsistency

      f[10..29] : per-probe flags (as before)
      
      ★ NEW:
      f[30] : SYN done
      f[31] : BANNER done
      f[32] : OS done
      f[33] : SERVICE done
    """

    f = features[host_idx]

    # ----------------------
    # Basic counts
    # ----------------------
    f[0] += 1
    n = f[0]

    if result.port_open is True:
        f[1] += 1
    elif result.port_open is False:
        f[2] += 1

    # Running RTT mean + variance
    old_mean = f[3]
    new_mean = old_mean + (result.rtt - old_mean) / n
    f[3] = new_mean

    if n == 1:
        f[4] = 0.0
    else:
        f[4] += (result.rtt - old_mean) * (result.rtt - new_mean)

    # Banner / OS scores
    f[5] = max(f[5], result.banner_score)
    f[6] = max(f[6], result.os_guess_score)

    # Honeypot artefacts
    if result.honeypot_artefact_flag:
        f[7] += 1

    # Probe diversity
    base = 10
    idx = base + probe_type
    if idx < 30:
        if f[idx] == 0:
            f[8] += 1
        f[idx] = 1

    # Inconsistency
    if f[5] > 0 and f[6] > 0:
        f[9] = abs(f[5] - f[6])

    # ----------------------------------------------------------
    # ★ NEW: explicit probe completion flags for PPO visibility
    # ----------------------------------------------------------
    # 1=SYN, 2=BANNER, 3=OS, 4=SERVICE
    if probe_type == 1: f[30] = 1.0
    if probe_type == 2: f[31] = 1.0
    if probe_type == 3: f[32] = 1.0
    if probe_type == 4: f[33] = 1.0

    features[host_idx] = f
