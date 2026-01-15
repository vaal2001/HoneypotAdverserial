import gymnasium as gym
import numpy as np

from .host_profiles import HostType, DEVICE_TYPES
from .network_generator import generate_random_network
from .detectability import (compute_real_baseline_stats, detectability_for_hosts)

def _clip_z(z, z_max = 4.0):
    return float(min(max(z, 0.0), z_max))

def detectability_single(host, degree, baseline):
    """
    Heuristic detectability score for a single host (0..~10).
    Meant as a conventional attacker heuristic.
    Lower is beter (less detectable).
    """
    rm = host.response_model

    z_rtt_std = _clip_z(abs(rm.rtt_std - baseline.rtt_std_mean) / baseline.rtt_std_std)
    z_banner = _clip_z(abs(rm.banner_noise - baseline.banner_noise_mean) / baseline.banner_noise_std)
    z_os = _clip_z(abs(rm.os_fingerprint - baseline.os_fp_mean) / baseline.os_fp_std)
    z_svc = _clip_z(abs(rm.service_count - baseline.service_count_mean) / baseline.service_count_std)
    z_deg = _clip_z(abs(degree - baseline.degree_mean) / baseline.degree_std)

    low_jitter = max(0.0, (baseline.rtt_std_mean - rm.rtt_std) / (baseline.rtt_std_mean + 1e-3))
    low_jitter_penalty = min(low_jitter, 2.0)  # cap

    artefact_penalty = min(rm.artefact_prob * 5.0, 3.0)

    score = (
        0.8 * z_rtt_std +
        0.6 * z_banner +
        1.0 * z_os +
        0.4 * z_svc +
        0.4 * z_deg +
        1.0 * low_jitter_penalty +
        1.5 * artefact_penalty
    )

    score = float(min(max(score, 0.0), 10.0))
    return score


def heuristic_attack_metrics(env):
    """
    Heuristic attacker metrics (Attack Success Rate + Honey Detection Rate)
    """
    hosts = env.hosts
    N_actual = env.N_actual
    adj = env.adjacency
    baseline = env.baseline_stats

    if baseline is None or N_actual == 0:
        return 0.0, 0.0

    deg = adj[:N_actual, :N_actual].sum(axis=1).astype(float)

    scores = []
    labels = []

    for i, h in enumerate(hosts[:N_actual]):
        s = detectability_single(h, degree=deg[i], baseline=baseline)
        scores.append(s)
        labels.append(h.host_type == h.host_type.__class__.HONEYPOT)

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)

    tau = 3.0

    preds = scores >= tau

    true_honey = labels
    true_real = ~labels

    TP = np.logical_and(preds, true_honey).sum()
    FN = np.logical_and(~preds, true_honey).sum()
    TN = np.logical_and(~preds, true_real).sum()
    FP = np.logical_and(preds, true_real).sum()

    N = max(1, N_actual)
    asr = float((TP + TN) / N)
    hdr = float(TP / max(1, (TP + FN)))

    return asr, hdr


class HoneypotDefenderEnv(gym.Env):
    """
    Simple defender environment (no attacker).

    Observation (Dict):
        node_features:   (N_max, F)
        adjacency:       (N_max, N_max)
        remaining_budget:(1,)

    Actions:
        Discrete(N_max * K), K = 8, per host:
            0: increase RTT std (meer jitter richting real)
            1: decrease RTT std (minder jitter richting real)
            2: increase banner noise
            3: decrease banner noise
            4: increase artefact prob   (slechte actie voor defender, maar toegestaan)
            5: decrease artefact prob
            6: adjust service_count richting typische waarde
            7: adjust os_fingerprint richting 'real' profiel
    """

    metadata = {"render_modes": []}

    def __init__(self, N_max = 40, max_steps = 100, seed = None):
        super().__init__()
        self.N_max = N_max
        self.max_steps = max_steps
        self.K = 8  # actions per host

        self.rng = np.random.default_rng(seed)

        self.N_actual = 0
        self.hosts = []
        self.adjacency = np.zeros((N_max, N_max), dtype=np.float32)
        self.baseline_stats = None
        self.step_count = 0

        self.baseline_detect_honey: float = 0.0
        self.baseline_detect_real: float = 0.0

        self.last_mean_detect_honey: float = 0.0
        self.last_mean_detect_real: float = 0.0

        self.F = 1 + len(DEVICE_TYPES) + 7

        self.observation_space = gym.spaces.Dict(
            {
                "node_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N_max, self.F), dtype=np.float32),
                "adjacency": gym.spaces.Box(low=0.0, high=1.0, shape=(N_max, N_max), dtype=np.float32),
                "remaining_budget": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            }
        )

        self.action_space = gym.spaces.Discrete(N_max * self.K)

        self.episode_reward = 0.0

    def _build_node_features(self):
        """
        Build (N_max, F) node feature matrix.
        """
        feats = np.zeros((self.N_max, self.F), dtype=np.float32)

        if self.N_actual == 0:
            return feats

        deg = self.adjacency[:self.N_actual, :self.N_actual].sum(axis=1).astype(float)

        for i, h in enumerate(self.hosts[: self.N_actual]):
            rm = h.response_model
            f = []

            f.append(1.0 if h.host_type == HostType.HONEYPOT else 0.0)

            for dt in DEVICE_TYPES:
                f.append(1.0 if h.device_type == dt else 0.0)

            rtt_mean_norm = np.clip(rm.rtt_mean, 1.0, 200.0) / 200.0
            rtt_std_norm = np.clip(rm.rtt_std, 0.5, 40.0) / 40.0
            banner_norm = np.clip(rm.banner_noise, 0.0, 1.0)
            os_norm = np.clip(rm.os_fingerprint, 0.3, 1.5) / 1.5
            artefact_norm = np.clip(rm.artefact_prob, 0.0, 1.0)
            svc_norm = np.clip(rm.service_count, 1, 10) / 10.0
            degree_norm = deg[i] / max(1.0, self.N_actual - 1)

            f.extend([rtt_mean_norm, rtt_std_norm, banner_norm, os_norm, artefact_norm, svc_norm, degree_norm])

            feats[i, :] = np.asarray(f, dtype=np.float32)

        return feats

    def _build_observation(self):
        remaining = max(self.max_steps - self.step_count, 0)
        remaining_norm = np.array([remaining / self.max_steps], dtype=np.float32)

        node_features = self._build_node_features()

        return {"node_features": node_features, "adjacency": self.adjacency.astype(np.float32), "remaining_budget": remaining_norm}

    def flatten_obs(self, obs):
        nf = obs["node_features"].reshape(-1)
        adj = obs["adjacency"].reshape(-1)
        rb = obs["remaining_budget"].reshape(-1)
        return np.concatenate([nf, adj, rb], axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed = None, options = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.N_actual, self.hosts, adj_padded = generate_random_network(self.N_max, self.rng)
        self.adjacency[:, :] = adj_padded.astype(np.float32)

        self.baseline_stats = compute_real_baseline_stats(self.hosts, self.adjacency, self.N_actual)

        self.step_count = 0

        assert self.baseline_stats is not None
        mean_h, mean_r = detectability_for_hosts(self.hosts, self.adjacency, self.baseline_stats, self.N_actual)
        self.last_mean_detect_honey = mean_h
        self.last_mean_detect_real = mean_r
        self.baseline_detect_honey = mean_h
        self.baseline_detect_real = mean_r

        self.episode_reward = 0.0

        obs = self._build_observation()
        info = {"mean_detect_honeypot": mean_h, "mean_detect_real": mean_r}
        return obs, info

    def _decode_action(self, action):
        host_idx = action // self.K
        sub = action % self.K
        return host_idx, sub

    def _apply_action(self, host_idx, sub_action):
        if not (0 <= host_idx < self.N_actual):
            return

        host = self.hosts[host_idx]
        rm = host.response_model

        def lerp(current, target, alpha):
            return (1.0 - alpha) * current + alpha * target

        is_honey = (host.host_type == HostType.HONEYPOT)

        # ---- 0 / 1: RTT jitter ----
        if sub_action == 0:  # increase RTT std
            if self.baseline_stats:
                target = self.baseline_stats.rtt_std_mean
            else:
                target = rm.rtt_std + 5.0
            alpha = 0.4 if is_honey else 0.2
            rm.rtt_std = lerp(rm.rtt_std, target, alpha)

        elif sub_action == 1:  # decrease RTT std
            if self.baseline_stats:
                target = self.baseline_stats.rtt_std_mean
            else:
                target = max(rm.rtt_std - 5.0, 0.5)
            alpha = 0.4 if rm.rtt_std > target else 0.2
            rm.rtt_std = lerp(rm.rtt_std, target, alpha)

        # ---- 2 / 3: banner noise ----
        elif sub_action == 2:  # increase banner noise
            if self.baseline_stats:
                target = max(self.baseline_stats.banner_noise_mean, rm.banner_noise + 0.05)
            else:
                target = rm.banner_noise + 0.1
            alpha = 0.4 if is_honey else 0.2
            rm.banner_noise = lerp(rm.banner_noise, target, alpha)

        elif sub_action == 3:  # decrease banner noise
            if self.baseline_stats:
                target = self.baseline_stats.banner_noise_mean
            else:
                target = max(rm.banner_noise - 0.1, 0.0)
            alpha = 0.4 if rm.banner_noise > target else 0.2
            rm.banner_noise = lerp(rm.banner_noise, target, alpha)

        # ---- 4 / 5: artefact_prob ----
        elif sub_action == 4:  # increase artefact prob
            rm.artefact_prob += 0.05

        elif sub_action == 5:  # decrease artefact prob
            alpha = 0.6 if is_honey else 0.3
            target = 0.0
            rm.artefact_prob = lerp(rm.artefact_prob, target, alpha)

        # ---- 6: service_count ----
        elif sub_action == 6:
            target = 3
            if rm.service_count < target:
                rm.service_count += 1
            elif rm.service_count > target:
                rm.service_count -= 1

        # ---- 7: os_fingerprint ----
        elif sub_action == 7:
            target = 1.0
            alpha = 0.5 if is_honey else 0.2
            rm.os_fingerprint = lerp(rm.os_fingerprint, target, alpha)

        rm.rtt_std = float(np.clip(rm.rtt_std, 0.5, 40.0))
        rm.rtt_mean = float(np.clip(rm.rtt_mean, 1.0, 200.0))
        rm.banner_noise = float(np.clip(rm.banner_noise, 0.0, 1.0))
        rm.os_fingerprint = float(np.clip(rm.os_fingerprint, 0.3, 1.5))
        rm.artefact_prob = float(np.clip(rm.artefact_prob, 0.0, 1.0))
        rm.service_count = int(np.clip(rm.service_count, 1, 10))

        host.response_model = rm

    def step(self, action: int):
        self.step_count += 1

        prev_h = self.last_mean_detect_honey

        host_idx, sub = self._decode_action(action)
        self._apply_action(host_idx, sub)

        assert self.baseline_stats is not None
        mean_h, mean_r = detectability_for_hosts(self.hosts, self.adjacency, self.baseline_stats, self.N_actual)
        self.last_mean_detect_honey = mean_h
        self.last_mean_detect_real = mean_r

        improvement = prev_h - mean_h

        real_excess = max(0.0, mean_r - self.baseline_detect_real)
        real_penalty = 0.3 * real_excess

        step_cost = -0.01

        reward = improvement - real_penalty + step_cost

        terminated = self.step_count >= self.max_steps
        truncated = False

        if terminated or truncated:
            self.episode_reward += reward

            asr, hrd = heuristic_attack_metrics(self)

            print(f"{self.episode_reward:.2f},{asr:.2f},{hrd:.2f}")

        obs = self._build_observation()
        info = {"mean_detect_honeypot": mean_h, "mean_detect_real": mean_r,}
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Step {self.step_count}:")
        print(f"  mean_detect_honeypot = {self.last_mean_detect_honey:.3f}")
        print(f"  mean_detect_real     = {self.last_mean_detect_real:.3f}")
        print(f"  baseline_honey       = {self.baseline_detect_honey:.3f}")
        print(f"  baseline_real        = {self.baseline_detect_real:.3f}")
