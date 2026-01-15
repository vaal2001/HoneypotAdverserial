import gymnasium as gym
import numpy as np

from .shared_host_profiles import HostType
from .shared_network_generator import generate_random_network, DEVICE_TYPES
from .shared_probe_models import (simulate_ping, simulate_syn_scan, simulate_banner, simulate_os, simulate_service)
from .attacker_feature_update import init_feature_matrix, update_features
from .attacker_rewards import (attacker_reward as reward_classification, step_cost)
from .defender_rewards import defender_reward

class MultiAgentHoneypotEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__( self, N_max = 40, F_attacker = 34, defender_F = 1 + 4 + 7, max_attacker_actions = 200, seed = None):
        super().__init__()

        self.N_max = N_max
        self.F_attacker = F_attacker
        self.F_defender = defender_F
        self.K_attacker = 8
        self.K_defender = 8
        self.max_attacker_actions = max_attacker_actions
        self.rng = np.random.default_rng(seed)

        self.N_actual = 0
        self.hosts = []
        self.adj_true = np.zeros((N_max, N_max), dtype=int)

        self.node_features = init_feature_matrix(N_max, F_attacker)
        self.adj_discovered = np.zeros((N_max, N_max), dtype=int)
        self.mask_known = np.zeros(N_max, dtype=np.int8)
        self.mask_classified = np.zeros(N_max, dtype=np.int8)
        self.classified_as = np.full(N_max, None, dtype=object)
        self.probes_done = np.zeros((N_max, 4), dtype=np.int8)
        self.ping_done = np.zeros(N_max, dtype=np.int8)
        self.attacker_actions_used = 0

        self.observation_space = gym.spaces.Dict(
            {
                "attacker": gym.spaces.Dict(
                    {
                        "node_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N_max, F_attacker), dtype=np.float32),
                        "adjacency": gym.spaces.Box(low=0.0, high=1.0, shape=(N_max, N_max), dtype=np.float32),
                        "mask_known": gym.spaces.MultiBinary(N_max),
                        "mask_classified": gym.spaces.MultiBinary(N_max),
                        "remaining_budget": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                        "action_mask": gym.spaces.MultiBinary(N_max * self.K_attacker)
                    }
                ),
                "defender": gym.spaces.Dict(
                    {
                        "node_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N_max, defender_F), dtype=np.float32),
                        "adjacency": gym.spaces.Box(low=0.0, high=1.0, shape=(N_max, N_max), dtype=np.float32),
                        "remaining_budget": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
                    }
                ),
            }
        )

        self.action_spaces = {"attacker": gym.spaces.Discrete(N_max * self.K_attacker), "defender": gym.spaces.Discrete(N_max * self.K_defender)}
        self.n_classified = 0
        self.n_current_classified = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.N_actual, self.hosts, self.adj_true = generate_random_network(self.N_max, self.rng)

        self._reset_attacker_state()

        obs_att = self._build_attacker_obs()
        obs_def = self._build_defender_obs()

        return {"attacker": obs_att, "defender": obs_def}, {}

    def _reset_attacker_state(self):
        self.node_features = init_feature_matrix(self.N_max, self.F_attacker)
        self.adj_discovered[:] = 0
        self.mask_known[:] = 0
        self.mask_classified[:] = 0
        self.classified_as[:] = None
        self.probes_done[:, :] = 0
        self.ping_done[:] = 0
        self.attacker_actions_used = 0
        self.n_classified = 0
        self.n_current_classified = 0

    def _build_attacker_obs(self):
        remaining = max(self.max_attacker_actions - self.attacker_actions_used, 0)
        remaining_norm = np.array([remaining / self.max_attacker_actions], dtype=np.float32)

        return {
            "node_features": self.node_features.astype(np.float32),
            "adjacency": self.adj_discovered.astype(np.float32),
            "mask_known": self.mask_known.astype(np.int8),
            "mask_classified": self.mask_classified.astype(np.int8),
            "remaining_budget": remaining_norm,
            "action_mask": self._build_attacker_action_mask(),
        }

    def _build_defender_obs(self):
        feats = np.zeros((self.N_max, self.F_defender), dtype=np.float32)

        if self.N_actual > 0:
            deg = self.adj_true[:self.N_actual, :self.N_actual].sum(axis=1).astype(float)

            for i, h in enumerate(self.hosts[: self.N_actual]):
                rm = h.response_model
                f = []

                f.append(1.0 if h.host_type == HostType.HONEYPOT else 0.0)

                for dt in DEVICE_TYPES:
                    f.append(1.0 if h.device_type == dt else 0.0)

                rtt_mean_norm = rm.rtt_mean / 200.0
                rtt_std_norm = rm.rtt_std / 40.0
                banner_norm = rm.banner_noise
                os_norm = rm.os_fingerprint / 1.5
                artefact_norm = rm.artefact_prob
                svc_norm = rm.service_count / 10.0
                degree_norm = deg[i] / max(1, self.N_actual - 1)

                f.extend([rtt_mean_norm, rtt_std_norm, banner_norm, os_norm, artefact_norm, svc_norm, degree_norm])

                feats[i, :] = np.asarray(f, dtype=np.float32)

        remaining = max(self.max_attacker_actions - self.attacker_actions_used, 0)
        remaining_norm = np.array([remaining / self.max_attacker_actions], dtype=np.float32)

        return {"node_features": feats.astype(np.float32), "adjacency": self.adj_true.astype(np.float32), "remaining_budget": remaining_norm}

    def _build_attacker_action_mask(self):
        mask = np.zeros(self.N_max * self.K_attacker, dtype=bool)

        for h in range(self.N_max):
            if h >= self.N_actual:
                continue

            start = h * self.K_attacker

            known = self.mask_known[h]
            classified = self.mask_classified[h]
            ping_done = self.ping_done[h]

            f = self.node_features[h]
            banner_done = (f[31] == 1.0)
            os_done = (f[32] == 1.0)

            if classified:
                continue

            if not known:
                if not ping_done:
                    mask[start + 0] = True
                mask[start + 1] = mask[start + 2] = mask[start + 3] = mask[start + 4] = True
                mask[start + 7] = True
                continue

            if not ping_done:
                mask[start + 0] = True

            mask[start + 1] = mask[start + 2] = mask[start + 3] = mask[start + 4] = True

            if banner_done and os_done:
                mask[start + 5] = True
                mask[start + 6] = True

            mask[start + 7] = True

        return mask

    def step(self, actions):

        defender_action = actions["defender"]
        attacker_action = actions["attacker"]

        self._apply_defender_action(defender_action)
        att_reward = self._apply_attacker_action(attacker_action)
        def_reward = defender_reward(att_reward)

        terminated = False
        truncated = False

        if self.attacker_actions_used >= self.max_attacker_actions:
            terminated = True

        if self.mask_classified[:self.N_actual].sum() == self.N_actual:
            terminated = True

        obs = {"attacker": self._build_attacker_obs(), "defender": self._build_defender_obs()}

        rewards = {"attacker": att_reward, "defender": def_reward}

        if terminated:
            print(att_reward)
            acc = self.n_current_classified/self.n_classified
            print(f"{att_reward:.2f},{acc:.2f}")

        return obs, rewards, terminated, truncated, {}

    def _apply_defender_action(self, action):
        host_idx = action // self.K_defender
        sub = action % self.K_defender

        if not (0 <= host_idx < self.N_actual):
            return

        h = self.hosts[host_idx]
        rm = h.response_model

        def lerp(a, b, alpha):
            return (1 - alpha) * a + alpha * b

        if sub == 0:
            rm.rtt_std = lerp(rm.rtt_std, rm.rtt_std + 5.0, 0.5)
        elif sub == 1:
            rm.rtt_std = lerp(rm.rtt_std, max(0.5, rm.rtt_std - 5.0), 0.5)
        elif sub == 2:
            rm.banner_noise = lerp(rm.banner_noise, rm.banner_noise + 0.1, 0.5)
        elif sub == 3:
            rm.banner_noise = lerp(rm.banner_noise, max(0.0, rm.banner_noise - 0.1), 0.5)
        elif sub == 4:
            rm.artefact_prob = min(1.0, rm.artefact_prob + 0.05)
        elif sub == 5:
            rm.artefact_prob = lerp(rm.artefact_prob, 0.0, 0.6)
        elif sub == 6:
            if rm.service_count < 5:
                rm.service_count += 1
        elif sub == 7:
            rm.os_fingerprint = lerp(rm.os_fingerprint, 1.0, 0.5)

        rm.rtt_std = float(np.clip(rm.rtt_std, 0.5, 40.0))
        rm.banner_noise = float(np.clip(rm.banner_noise, 0.0, 2.0))
        rm.os_fingerprint = float(np.clip(rm.os_fingerprint, 0.3, 1.5))
        rm.artefact_prob = float(np.clip(rm.artefact_prob, 0.0, 1.0))
        rm.service_count = int(np.clip(rm.service_count, 1, 10))

    def _apply_attacker_action(self, action):
        mask = self._build_attacker_action_mask()
        if not mask[action]:
            return -1.0

        host_idx = action // self.K_attacker
        sub = action % self.K_attacker

        self.attacker_actions_used += 1

        reward = step_cost()

        if not (0 <= host_idx < self.N_actual):
            return reward - 1.0

        host = self.hosts[host_idx]

        if sub == 0:
            return reward + self._att_ping(host_idx)

        if sub == 1:
            return reward + self._att_syn(host_idx)

        if sub == 2:
            return reward + self._att_banner(host_idx)

        if sub == 3:
            return reward + self._att_os(host_idx)

        if sub == 4:
            return reward + self._att_service(host_idx)

        if sub == 5:
            self.mask_classified[host_idx] = 1
            self.classified_as[host_idx] = False

            self.n_classified += 1

            if host.host_type == HostType.REAL:
                self.n_current_classified += 1
            
            return reward + reward_classification(host, False)

        if sub == 6:
            self.mask_classified[host_idx] = 1
            self.classified_as[host_idx] = True

            self.n_classified += 1

            if host.host_type == HostType.HONEYPOT:
                self.n_current_classified += 1

            return reward + reward_classification(host, True)

        return reward

    def _att_ping(self, idx):
        if self.ping_done[idx] == 1:
            return 0.0

        self.ping_done[idx] = 1
        self.mask_known[idx] = 1

        res = simulate_ping(self.hosts[idx], self.rng)
        update_features(self.node_features, idx, 0, res)

        for n in range(self.N_actual):
            if self.adj_true[idx, n] == 1:
                self.mask_known[n] = 1
                self.adj_discovered[idx, n] = 1
                self.adj_discovered[n, idx] = 1

        return 0.0

    def _att_syn(self, idx):
        res = simulate_syn_scan(self.hosts[idx], self.rng)
        update_features(self.node_features, idx, 1, res)
        return 0.05 + (0.1 if res.artefact else 0.0)

    def _att_banner(self, idx):
        res = simulate_banner(self.hosts[idx], self.rng)
        update_features(self.node_features, idx, 2, res)
        return 0.20 + (0.1 if res.artefact else 0.0)

    def _att_os(self, idx):
        res = simulate_os(self.hosts[idx], self.rng)
        update_features(self.node_features, idx, 3, res)
        return 0.20

    def _att_service(self, idx):
        res = simulate_service(self.hosts[idx], self.rng)
        update_features(self.node_features, idx, 4, res)
        return 0.25 + (0.1 if res.artefact else 0.0)

    def flatten_attacker_obs(self, obs):
        nf = obs["node_features"].reshape(-1)
        adj = obs["adjacency"].reshape(-1)
        mk = obs["mask_known"].reshape(-1)
        mc = obs["mask_classified"].reshape(-1)
        rb = obs["remaining_budget"].reshape(-1)
        return np.concatenate([nf, adj, mk, mc, rb], axis=0)

    def flatten_defender_obs(self, obs):
        nf = obs["node_features"].reshape(-1)
        adj = obs["adjacency"].reshape(-1)
        rb = obs["remaining_budget"].reshape(-1)
        return np.concatenate([nf, adj, rb], axis=0)
