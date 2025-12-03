# adversarial_env.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import gymnasium as gym

# --- Attacker imports -------------------------------------------------
from HoneypotAttacker.env.env import HoneypotDetectionEnv
from HoneypotAttacker.env.network_generator import generate_random_network
from HoneypotAttacker.env.host_profiles import (
    Host as AttHost,
    HostType as AttHostType,
)

# --- Defender imports -------------------------------------------------
from HoneypotDefender.env.env import HoneypotDefenderEnv
from HoneypotDefender.env.host_profiles import (
    Host as DefHost,
    HostType as DefHostType,
    ResponseModel as DefRM,
)

# ----------------------------------------------------------------------
# Mapping helpers: attacker <-> defender hostprofielen
# ----------------------------------------------------------------------


def build_defender_hosts_from_attacker(att_hosts: List[AttHost]) -> List[DefHost]:
    """
    Maak defender-hosts op basis van attacker-hosts.

    Mapping:
      attacker.ResponseModel:
        - rtt_mean
        - rtt_std
        - noise_level
        - honeypot_artefact_prob
        - banner_template
      defender.ResponseModel:
        - rtt_mean
        - rtt_std
        - banner_noise
        - os_fingerprint
        - artefact_prob
        - service_count
    """
    def_hosts: List[DefHost] = []

    for h in att_hosts:
        is_honey = (h.host_type == AttHostType.HONEYPOT)

        rm_att = h.response_model

        rm_def = DefRM(
            rtt_mean=float(rm_att.rtt_mean),
            rtt_std=float(rm_att.rtt_std),
            banner_noise=float(rm_att.noise_level),
            os_fingerprint=0.7 if is_honey else 1.0,
            artefact_prob=float(rm_att.honeypot_artefact_prob),
            service_count=int(len(h.services)),
        )

        def_hosts.append(
            DefHost(
                host_id=h.host_id,
                host_type=DefHostType.HONEYPOT if is_honey else DefHostType.REAL,
                device_type=h.device_type,
                services=[],            # defender gebruikt services normaal niet expliciet
                response_model=rm_def,
            )
        )

    return def_hosts


def sync_attacker_from_defender(att_hosts: List[AttHost], def_hosts: List[DefHost]) -> None:
    """
    Kopieer relevante parameters terug naar attacker-hosts.

    Hierdoor ziet de attacker de door de defender aangepaste:
      - rtt_mean / rtt_std
      - banner_noise (→ noise_level)
      - artefact_prob (→ honeypot_artefact_prob)

    OS-fingerprint, service_count en eventuele andere dingen
    blijven puur defender-side.
    """
    for att, d in zip(att_hosts, def_hosts):
        rm_att = att.response_model
        rm_def = d.response_model

        rm_att.rtt_mean = float(rm_def.rtt_mean)
        rm_att.rtt_std = float(rm_def.rtt_std)
        rm_att.noise_level = float(rm_def.banner_noise)
        rm_att.honeypot_artefact_prob = float(rm_def.artefact_prob)

        # banner_template, services etc. laten we zoals ze waren
        att.response_model = rm_att


# ----------------------------------------------------------------------
# World state container
# ----------------------------------------------------------------------


@dataclass
class WorldState:
    N_actual: int
    adjacency: np.ndarray          # (N_max, N_max)
    attacker_hosts: List[AttHost]
    defender_hosts: List[DefHost]


# ----------------------------------------------------------------------
# Multi-agent adversarial env
# ----------------------------------------------------------------------


class HoneypotAdversarialEnv(gym.Env):
    """
    Adversarial omgeving met twee agents:

      - attacker:  HoneypotDetectionEnv
      - defender:  HoneypotDefenderEnv

    API:
      reset() -> (obs, info)
        obs: {
          "attacker": <obs_attacker>,
          "defender": <obs_defender>,
        }

      step({"attacker": a_att, "defender": a_def})
        -> (obs, reward, terminated, truncated, info)

        reward: {
          "attacker": r_att,
          "defender": r_def,
        }
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        N_max: int = 40,
        attacker_max_actions: int = 200,
        defender_max_steps: int = 100,
        seed: int | None = None,
    ):
        super().__init__()
        self.N_max = N_max
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Onderliggende single-agent envs
        self.attacker_env = HoneypotDetectionEnv(
            N_max=N_max,
            max_actions=attacker_max_actions,
            seed=seed,
        )
        self.defender_env = HoneypotDefenderEnv(
            N_max=N_max,
            max_steps=defender_max_steps,
            seed=seed,
        )

        self.world: WorldState | None = None

        # Optioneel: "multi-agent" spaces, handig als je zelf iets bovenop PPO bouwt
        self.observation_space = gym.spaces.Dict(
            {
                "attacker": self.attacker_env.observation_space,
                "defender": self.defender_env.observation_space,
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                "attacker": self.attacker_env.action_space,
                "defender": self.defender_env.action_space,
            }
        )

    # ------------------------------------------------------------------
    # World generation
    # ------------------------------------------------------------------
    def _generate_world(self) -> WorldState:
        """
        Gebruik de attacker-network generator als canonical wereld.
        """
        N_actual, att_hosts, adj_padded = generate_random_network(
            self.N_max, self.rng
        )
        def_hosts = build_defender_hosts_from_attacker(att_hosts)
        return WorldState(
            N_actual=N_actual,
            adjacency=adj_padded.astype(np.float32),
            attacker_hosts=att_hosts,
            defender_hosts=def_hosts,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.world = self._generate_world()
        w = self.world

        # Init onderliggende envs met dezelfde wereld
        obs_att, info_att = self.attacker_env.reset_with_world(
            w.N_actual, w.attacker_hosts, w.adjacency
        )
        obs_def, info_def = self.defender_env.reset_with_world(
            w.N_actual, w.defender_hosts, w.adjacency
        )

        obs = {
            "attacker": obs_att,
            "defender": obs_def,
        }
        info = {
            "attacker": info_att,
            "defender": info_def,
        }
        return obs, info

    def step(self, action: Dict[str, int]):
        """
        action: {"attacker": int, "defender": int}
        """
        assert self.world is not None, "Call reset() before step()."

        a_def = int(action.get("defender", 0))
        a_att = int(action.get("attacker", 0))

        # 1) Defender-actie eerst (vervormt de signatuur van de hosts)
        obs_def, r_def, term_def, trunc_def, info_def = self.defender_env.step(a_def)

        # 2) Defender → Attacker: sync de gewijzigde response-parameters terug
        sync_attacker_from_defender(
            self.world.attacker_hosts, self.world.defender_hosts
        )

        # 3) Attacker-actie op de aangepaste wereld
        obs_att, r_att, term_att, trunc_att, info_att = self.attacker_env.step(a_att)

        terminated = term_def or term_att
        truncated = trunc_def or trunc_att

        obs = {
            "attacker": obs_att,
            "defender": obs_def,
        }
        reward = {
            "attacker": r_att,
            "defender": r_def,
        }
        info = {
            "attacker": info_att,
            "defender": info_def,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        print("=== HoneypotAdversarialEnv ===")
        print(f"  Attacker steps: {self.attacker_env.step_count}")
        print(f"  Defender steps: {self.defender_env.step_count}")
