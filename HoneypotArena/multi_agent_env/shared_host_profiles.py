from dataclasses import dataclass
from enum import Enum, auto

class HostType(Enum):
    REAL = auto()
    HONEYPOT = auto()

@dataclass
class ResponseModel:
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
