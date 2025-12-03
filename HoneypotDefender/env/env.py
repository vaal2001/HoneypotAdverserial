from __future__ import annotations
from typing import List, Dict, Any

import gymnasium as gym
import numpy as np

from .host_profiles import Host, HostType, DEVICE_TYPES
from .network_generator import generate_random_network
from .detectability import (
    compute_real_baseline_stats,
    detectability_for_hosts,
    RealBaselineStats,
)


class HoneypotDefenderEnv(gym.Env):
    """
    Enkelvoudige defender-env (geen attacker).

    De defender kan de 'signatuur' van hosts aanpassen zodat honeypots
    moeilijker te detecteren zijn, terwijl echte hosts niet verdachter
    mogen worden.

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

    def __init__(
        self,
        N_max: int = 40,
        max_steps: int = 100,
        seed: int | None = None,
    ):
        super().__init__()
        self.N_max = N_max
        self.max_steps = max_steps
        self.K = 8  # actions per host

        self.rng = np.random.default_rng(seed)

        # dynamische state
        self.N_actual: int = 0
        self.hosts: List[Host] = []
        self.adjacency = np.zeros((N_max, N_max), dtype=np.float32)
        self.baseline_stats: RealBaselineStats | None = None
        self.step_count = 0

        # baseline detectability (bij reset, voor reward shaping)
        self.baseline_detect_honey: float = 0.0
        self.baseline_detect_real: float = 0.0

        # laatste detectability per stap (voor logging / evaluatie)
        self.last_mean_detect_honey: float = 0.0
        self.last_mean_detect_real: float = 0.0

        # Features: host_type flag + device one-hot + 7 numerieke features
        #  1  : is_honeypot
        #  |DEVICE_TYPES|
        #  7  : [rtt_mean, rtt_std, banner_noise, os_fp, artefact_prob,
        #         service_count, degree_norm]
        self.F = 1 + len(DEVICE_TYPES) + 7

        # spaces
        self.observation_space = gym.spaces.Dict(
            {
                "node_features": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(N_max, self.F),
                    dtype=np.float32,
                ),
                "adjacency": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(N_max, N_max),
                    dtype=np.float32,
                ),
                "remaining_budget": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = gym.spaces.Discrete(N_max * self.K)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _build_node_features(self) -> np.ndarray:
        """
        Bouw (N_max, F) features uit huidige hosts + adjacency.
        """
        feats = np.zeros((self.N_max, self.F), dtype=np.float32)

        if self.N_actual == 0:
            return feats

        deg = self.adjacency[:self.N_actual, :self.N_actual].sum(axis=1).astype(float)

        for i, h in enumerate(self.hosts[: self.N_actual]):
            rm = h.response_model
            f: List[float] = []

            # type-flag
            f.append(1.0 if h.host_type == HostType.HONEYPOT else 0.0)

            # device one-hot
            for dt in DEVICE_TYPES:
                f.append(1.0 if h.device_type == dt else 0.0)

            # genormaliseerde numerieke features
            rtt_mean_norm = np.clip(rm.rtt_mean, 1.0, 200.0) / 200.0
            rtt_std_norm = np.clip(rm.rtt_std, 0.5, 40.0) / 40.0
            banner_norm = np.clip(rm.banner_noise, 0.0, 1.0)
            os_norm = np.clip(rm.os_fingerprint, 0.3, 1.5) / 1.5
            artefact_norm = np.clip(rm.artefact_prob, 0.0, 1.0)
            svc_norm = np.clip(rm.service_count, 1, 10) / 10.0
            degree_norm = deg[i] / max(1.0, self.N_actual - 1)

            f.extend(
                [
                    rtt_mean_norm,
                    rtt_std_norm,
                    banner_norm,
                    os_norm,
                    artefact_norm,
                    svc_norm,
                    degree_norm,
                ]
            )

            feats[i, :] = np.asarray(f, dtype=np.float32)

        return feats

    def _build_observation(self) -> Dict[str, Any]:
        remaining = max(self.max_steps - self.step_count, 0)
        remaining_norm = np.array([remaining / self.max_steps], dtype=np.float32)

        node_features = self._build_node_features()

        return {
            "node_features": node_features,
            "adjacency": self.adjacency.astype(np.float32),
            "remaining_budget": remaining_norm,
        }

    def flatten_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        nf = obs["node_features"].reshape(-1)
        adj = obs["adjacency"].reshape(-1)
        rb = obs["remaining_budget"].reshape(-1)
        return np.concatenate([nf, adj, rb], axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # genereer nieuw netwerk
        self.N_actual, self.hosts, adj_padded = generate_random_network(
            self.N_max, self.rng
        )
        self.adjacency[:, :] = adj_padded.astype(np.float32)

        # compute real-baseline stats
        self.baseline_stats = compute_real_baseline_stats(
            self.hosts, self.adjacency, self.N_actual
        )

        self.step_count = 0

        # init detectability metrics (baseline vóór defender-acties)
        assert self.baseline_stats is not None
        mean_h, mean_r = detectability_for_hosts(
            self.hosts,
            self.adjacency,
            self.baseline_stats,
            self.N_actual,
        )
        self.last_mean_detect_honey = mean_h
        self.last_mean_detect_real = mean_r
        self.baseline_detect_honey = mean_h
        self.baseline_detect_real = mean_r

        obs = self._build_observation()
        info: Dict[str, Any] = {
            "mean_detect_honeypot": mean_h,
            "mean_detect_real": mean_r,
        }
        return obs, info

    def _decode_action(self, action: int) -> tuple[int, int]:
        host_idx = action // self.K
        sub = action % self.K
        return host_idx, sub

    def _apply_action(self, host_idx: int, sub_action: int) -> None:
        if not (0 <= host_idx < self.N_actual):
            return

        host = self.hosts[host_idx]
        rm = host.response_model

        # eenvoudige nudge helper
        def lerp(current: float, target: float, alpha: float) -> float:
            return (1.0 - alpha) * current + alpha * target

        is_honey = (host.host_type == HostType.HONEYPOT)

        # ---- 0 / 1: RTT jitter richting real-baseline ----
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

        # ---- 2 / 3: banner noise richting real-baseline ----
        elif sub_action == 2:  # increase banner noise
            if self.baseline_stats:
                target = max(
                    self.baseline_stats.banner_noise_mean,
                    rm.banner_noise + 0.05,
                )
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
        elif sub_action == 4:  # increase artefact prob (defender-fout)
            rm.artefact_prob += 0.05

        elif sub_action == 5:  # decrease artefact prob (goed)
            alpha = 0.6 if is_honey else 0.3
            target = 0.0
            rm.artefact_prob = lerp(rm.artefact_prob, target, alpha)

        # ---- 6: service_count richting typische waarde ----
        elif sub_action == 6:
            # je kunt hier later device-type-specifieke targets gebruiken
            target = 3
            if rm.service_count < target:
                rm.service_count += 1
            elif rm.service_count > target:
                rm.service_count -= 1

        # ---- 7: os_fingerprint richting 'real' ----
        elif sub_action == 7:
            target = 1.0
            alpha = 0.5 if is_honey else 0.2
            rm.os_fingerprint = lerp(rm.os_fingerprint, target, alpha)

        # clamps voor stabiliteit
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
        prev_r = self.last_mean_detect_real

        host_idx, sub = self._decode_action(action)
        self._apply_action(host_idx, sub)

        # recompute detectability met actuele host-profielen
        assert self.baseline_stats is not None
        mean_h, mean_r = detectability_for_hosts(
            self.hosts,
            self.adjacency,
            self.baseline_stats,
            self.N_actual,
        )
        self.last_mean_detect_honey = mean_h
        self.last_mean_detect_real = mean_r

        # ------------------------------------------------------------------
        # Reward design:
        #  - Beloning voor verbetering van honeypot-detectability
        #    (lager is beter → improvement = prev_h - mean_h)
        #  - Penalty als REAL detectability hoger wordt dan baseline
        #  - Kleine step-cost
        # ------------------------------------------------------------------
        improvement = prev_h - mean_h

        real_excess = max(0.0, mean_r - self.baseline_detect_real)
        real_penalty = 0.3 * real_excess

        step_cost = -0.01

        reward = improvement - real_penalty + step_cost

        terminated = self.step_count >= self.max_steps
        truncated = False

        obs = self._build_observation()
        info = {
            "mean_detect_honeypot": mean_h,
            "mean_detect_real": mean_r,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Step {self.step_count}:")
        print(f"  mean_detect_honeypot = {self.last_mean_detect_honey:.3f}")
        print(f"  mean_detect_real     = {self.last_mean_detect_real:.3f}")
        print(f"  baseline_honey       = {self.baseline_detect_honey:.3f}")
        print(f"  baseline_real        = {self.baseline_detect_real:.3f}")
