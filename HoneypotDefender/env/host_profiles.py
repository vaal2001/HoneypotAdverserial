from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import numpy as np


class HostType(Enum):
    REAL = auto()
    HONEYPOT = auto()


DEVICE_TYPES: List[str] = [
    "linux_web",
    "win_desktop",
    "iot_sensor",
    "db_server",
]


@dataclass
class ResponseModel:
    """
    Parameters die de 'netwerk-signatuur' van een host bepalen.
    Deze worden door de defender gemanipuleerd.
    """
    rtt_mean: float          # ms
    rtt_std: float           # ms
    banner_noise: float      # 0..1
    os_fingerprint: float    # ~1.0 = typisch real, ~0.7 = honeypot
    artefact_prob: float     # 0..1
    service_count: int       # aantal zichtbare services


@dataclass
class Host:
    host_id: int
    host_type: HostType
    device_type: str
    response_model: ResponseModel


def _typical_service_count(device_type: str, rng: np.random.Generator) -> int:
    if device_type == "linux_web":
        return int(rng.integers(1, 4))
    if device_type == "win_desktop":
        return int(rng.integers(1, 3))
    if device_type == "iot_sensor":
        return int(rng.integers(1, 2))
    if device_type == "db_server":
        return int(rng.integers(1, 4))
    return int(rng.integers(1, 3))


def generate_random_host(
    host_id: int,
    is_honeypot: bool,
    rng: np.random.Generator,
) -> Host:
    """
    Genereer een hostprofiel.

    Real hosts:
      - hogere RTT-variatie
      - lage artefact_prob
      - os_fingerprint ~ 1.0

    Honeypots:
      - lagere RTT-variatie (uniformer)
      - hogere artefact_prob
      - os_fingerprint ~ 0.7
    """
    device_type = rng.choice(DEVICE_TYPES)
    service_count = _typical_service_count(device_type, rng)

    if not is_honeypot:
        # REAL
        rtt_mean = float(rng.uniform(10.0, 80.0))
        rtt_std = float(rng.uniform(5.0, 20.0))
        banner_noise = float(rng.uniform(0.05, 0.25))
        os_fingerprint = float(rng.normal(1.0, 0.1))
        artefact_prob = float(rng.uniform(0.0, 0.02))
    else:
        # HONEYPOT (initieel wat verdachter)
        rtt_mean = float(rng.uniform(10.0, 80.0))
        rtt_std = float(rng.uniform(1.0, 8.0))     # uniformer
        banner_noise = float(rng.uniform(0.0, 0.2))
        os_fingerprint = float(rng.normal(0.7, 0.1))
        artefact_prob = float(rng.uniform(0.1, 0.4))

    rm = ResponseModel(
        rtt_mean=rtt_mean,
        rtt_std=rtt_std,
        banner_noise=banner_noise,
        os_fingerprint=os_fingerprint,
        artefact_prob=artefact_prob,
        service_count=service_count,
    )

    host_type = HostType.HONEYPOT if is_honeypot else HostType.REAL

    return Host(
        host_id=host_id,
        host_type=host_type,
        device_type=device_type,
        response_model=rm,
    )
