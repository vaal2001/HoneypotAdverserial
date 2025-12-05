# shared_host_profiles.py
from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import numpy as np

class HostType(Enum):
    REAL = auto()
    HONEYPOT = auto()

@dataclass
class ResponseModel:
    """
    Unified response model used by both attacker & defender.
    Defender modifies these values.
    Attacker probes read these values.
    """
    rtt_mean: float
    rtt_std: float
    banner_noise: float
    os_fingerprint: float
    artefact_prob: float
    service_count: int

@dataclass
class Host:
    host_id: int
    host_type: HostType
    device_type: str
    response_model: ResponseModel
