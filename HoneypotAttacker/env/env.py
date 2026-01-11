import numpy as np
import gymnasium as gym
import time

from .network_generator import generate_random_network
from .host_profiles import HostType
from .probe_models import (simulate_ping, simulate_syn_scan, simulate_banner_grab, simulate_os_probe, simulate_service_probe)
from .feature_extractor import init_feature_matrix, update_features
from .observation_builder import build_observation
from .reward_models import (classification_reward, step_cost, honeypot_trigger_penalty, budget_exhausted_penalty, probe_reward_for_type, intrinsic_probe_signal, early_finish_bonus)

class HoneypotDetectionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, N_max = 40, F = 34, max_actions = 200, seed = None):
        super().__init__()
        self.seed = int(time.time())
        self.N_max = N_max
        self.F = F
        self.K = 8
        self.max_actions = max_actions

        self.rng = np.random.default_rng(seed)

        self.observation_space = gym.spaces.Dict(
            {
                "node_features": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N_max, F), dtype=np.float32),
                "adjacency": gym.spaces.Box(low=0.0, high=1.0, shape=(N_max, N_max), dtype=np.float32),
                "mask_known": gym.spaces.MultiBinary(N_max),
                "mask_classified": gym.spaces.MultiBinary(N_max),
                "remaining_budget": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "action_mask": gym.spaces.MultiBinary(N_max * self.K),
            }
        )

        self.action_space = gym.spaces.Discrete(N_max * self.K)

        self.N_actual = 0
        self.hosts = []
        self.adj_true = np.zeros((N_max, N_max), dtype=int)
        self.adj_discovered = np.zeros((N_max, N_max), dtype=int)

        self.node_features = init_feature_matrix(N_max, F)
        self.mask_known = np.zeros(N_max, dtype=np.int8)
        self.mask_classified = np.zeros(N_max, dtype=np.int8)
        self.classified_as = np.full(N_max, None, dtype=object)

        self.probes_done = np.zeros((N_max, 4), dtype=np.int8)

        self.ping_done = np.zeros(N_max, dtype=np.int8)

        self.actions_used = 0

        self.episode_reward = 0.0
        self.correct_classifications = 0
        self.total_classifications = 0

    def _generate_network(self):
        self.N_actual, self.hosts, self.adj_true = generate_random_network(self.N_max, self.rng)
        self.adj_discovered[:] = 0

    def _reset_agent_state(self):
        self.node_features = init_feature_matrix(self.N_max, self.F)
        self.mask_known[:] = 0
        self.mask_classified[:] = 0
        self.classified_as[:] = None
        self.actions_used = 0
        self.probes_done[:, :] = 0
        self.ping_done[:] = 0
        self.episode_reward = 0.0
        self.correct_classifications = 0
        self.total_classifications = 0

    def _decode_action(self, action):
        return action // self.K, action % self.K

    def _valid_host(self, host_idx):
        return 0 <= host_idx < self.N_actual

    def _build_action_mask(self):
        mask = np.zeros(self.N_max * self.K, dtype=bool)

        for h in range(self.N_max):
            start = h * self.K

            if h >= self.N_actual:
                continue

            known = self.mask_known[h]
            classified = self.mask_classified[h]
            ping_done = self.ping_done[h]

            f = self.node_features[h]
            banner_done = f[31] == 1.0
            os_done = f[32] == 1.0

            if classified:
                continue

            if not known:
                if not ping_done:
                    mask[start + 0] = True

                mask[start + 1] = True
                mask[start + 2] = True
                mask[start + 3] = True
                mask[start + 4] = True

                mask[start + 7] = True

                continue

            if not ping_done:
                mask[start + 0] = True

            mask[start + 1] = True
            mask[start + 2] = True
            mask[start + 3] = True
            mask[start + 4] = True

            if banner_done and os_done:
                mask[start + 5] = True
                mask[start + 6] = True

            mask[start + 7] = True

        return mask
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            seed = int(time.time())
            self.rng = np.random.default_rng(seed)

        self._generate_network()
        self._reset_agent_state()

        obs = build_observation(self.node_features, self.adj_discovered, self.mask_known, self.mask_classified, self.max_actions, self.max_actions)

        obs["action_mask"] = self._build_action_mask()
        return obs, {}

    def step(self, action):
        action_mask = self._build_action_mask()
        if not action_mask[action]:
            return (self._illegal_action_step(), -1.0, False, False, {})

        host_idx, action_type = self._decode_action(action)

        reward = step_cost()
        terminated = False
        truncated = False

        self.actions_used += 1

        if not self._valid_host(host_idx):
            reward -= 1.0
        else:
            reward += self._apply_action(host_idx, action_type)

        if self.actions_used >= self.max_actions:
            uncls = int(self.N_actual - self.mask_classified[: self.N_actual].sum())
            reward += budget_exhausted_penalty(uncls, self.N_actual)
            truncated = True

        if self.mask_classified[: self.N_actual].sum() == self.N_actual:
            terminated = True

        if terminated:
            remaining = max(self.max_actions - self.actions_used, 0)
            reward += early_finish_bonus(remaining, self.max_actions)
            self.episode_reward += reward

            accuracy = self.correct_classifications / self.total_classifications if self.total_classifications > 0 else 0.0
            print(f"{self.actions_used},{self.episode_reward:.2f},{accuracy:.2f}")

        obs = build_observation(self.node_features, self.adj_discovered, self.mask_known, self.mask_classified, self.max_actions - self.actions_used, self.max_actions)
        obs["action_mask"] = self._build_action_mask()
        return obs, reward, terminated, truncated, {}

    def _illegal_action_step(self):
        obs = build_observation(self.node_features, self.adj_discovered, self.mask_known, self.mask_classified, self.max_actions - self.actions_used, self.max_actions)
        obs["action_mask"] = self._build_action_mask()
        return obs

    def _apply_action(self, host_idx, action_type):
        host = self.hosts[host_idx]
        bonus = 0.0

        if action_type == 0:
            if self.ping_done[host_idx] == 1:
                return bonus

            self.ping_done[host_idx] = 1
            self.mask_known[host_idx] = 1
            result = simulate_ping(host, self.rng)
            update_features(self.node_features, host_idx, 0, result)

            for n in range(self.N_actual):
                if self.adj_true[host_idx, n] == 1:
                    self.mask_known[n] = 1
                    self.adj_discovered[host_idx, n] = 1
                    self.adj_discovered[n, host_idx] = 1

        elif action_type == 1:  # SYN_SCAN
            self.probes_done[host_idx, 0] = 1
            result = simulate_syn_scan(host, host.services[0].port, self.rng)
            update_features(self.node_features, host_idx, 1, result)
            bonus += probe_reward_for_type(1)
            bonus += intrinsic_probe_signal(host, result)

        elif action_type == 2:  # BANNER_GRAB
            self.probes_done[host_idx, 1] = 1
            result = simulate_banner_grab(host, self.rng)
            update_features(self.node_features, host_idx, 2, result)
            bonus += probe_reward_for_type(2)
            bonus += intrinsic_probe_signal(host, result)

        elif action_type == 3:  # OS_PROBE
            self.probes_done[host_idx, 2] = 1
            result = simulate_os_probe(host, self.rng)
            update_features(self.node_features, host_idx, 3, result)
            bonus += probe_reward_for_type(3)
            bonus += intrinsic_probe_signal(host, result)

        elif action_type == 4:  # SERVICE_PROBE
            self.probes_done[host_idx, 3] = 1
            result = simulate_service_probe(host, self.rng)
            update_features(self.node_features, host_idx, 4, result)
            bonus += probe_reward_for_type(4)
            bonus += intrinsic_probe_signal(host, result)

            if host.host_type == HostType.HONEYPOT and getattr(
                result, "honeypot_artefact_flag", False
            ):
                bonus += honeypot_trigger_penalty()

        elif action_type == 5:  # classify REAL
            self.mask_classified[host_idx] = 1
            self.classified_as[host_idx] = False

            self.total_classifications += 1
            correct = (host.host_type != HostType.HONEYPOT)
            if correct:
                self.correct_classifications += 1

            bonus += classification_reward(host, False)

        elif action_type == 6:  # classify HONEYPOT
            self.mask_classified[host_idx] = 1
            self.classified_as[host_idx] = True

            self.total_classifications += 1
            correct = (host.host_type == HostType.HONEYPOT)
            if correct:
                self.correct_classifications += 1

            bonus += classification_reward(host, True)

        return bonus

    def flatten_obs(self, obs):
        nf = obs["node_features"].reshape(-1)
        adj = obs["adjacency"].reshape(-1)
        mk = obs["mask_known"].reshape(-1)
        mc = obs["mask_classified"].reshape(-1)
        rb = obs["remaining_budget"].reshape(-1)
        return np.concatenate([nf, adj, mk, mc, rb], axis=0)

    def render(self):
        print("Known:", np.where(self.mask_known == 1)[0])
        print("Classified:", np.where(self.mask_classified == 1)[0])