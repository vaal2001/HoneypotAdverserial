import numpy as np

def init_feature_matrix(N_max, F):
    return np.zeros((N_max, F), dtype=np.float32)

def update_features(features, host_idx, probe_type, result):
    f = features[host_idx]

    f[0] += 1
    n = f[0]

    if result.port_open is True:
        f[1] += 1
    elif result.port_open is False:
        f[2] += 1

    old_mean = f[3]
    new_mean = old_mean + (result.rtt - old_mean) / n
    f[3] = new_mean

    if n == 1:
        f[4] = 0.0
    else:
        f[4] += (result.rtt - old_mean) * (result.rtt - new_mean)

    f[5] = max(f[5], result.banner_score)
    f[6] = max(f[6], result.os_guess_score)

    if result.artefact:
        f[7] += 1

    base = 10
    idx = base + probe_type
    if idx < 30:
        if f[idx] == 0:
            f[8] += 1
        f[idx] = 1

    if f[5] > 0 and f[6] > 0:
        f[9] = abs(f[5] - f[6])

    if probe_type == 1: f[30] = 1.0
    if probe_type == 2: f[31] = 1.0
    if probe_type == 3: f[32] = 1.0
    if probe_type == 4: f[33] = 1.0

    features[host_idx] = f
