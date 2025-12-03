from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import numpy as np


# ---------------------------------------------------------
# Host types
# ---------------------------------------------------------
class HostType(Enum):
    REAL = auto()
    HONEYPOT = auto()


# ---------------------------------------------------------
# Service running on a host
# ---------------------------------------------------------
@dataclass
class Service:
    port: int
    protocol: str
    service_type: str


# ---------------------------------------------------------
# Response model for probes
# ---------------------------------------------------------
@dataclass
class ResponseModel:
    rtt_mean: float
    rtt_std: float
    banner_template: str
    noise_level: float
    honeypot_artefact_prob: float


# ---------------------------------------------------------
# Host definition
# ---------------------------------------------------------
@dataclass
class Host:
    host_id: int
    host_type: HostType
    device_type: str
    services: List[Service]
    response_model: ResponseModel


# ---------------------------------------------------------
# Device types used for random host generation
# ---------------------------------------------------------
DEVICE_TYPES = [
    "linux_web",
    "win_desktop",
    "iot_sensor",
    "db_server",
]


# ---------------------------------------------------------
# Generate random services based on device type
# ---------------------------------------------------------
def _random_services(device_type: str, rng: np.random.Generator) -> List[Service]:
    services: List[Service] = []

    if device_type == "linux_web":
        services.append(Service(80, "tcp", "http"))
        if rng.random() < 0.5:
            services.append(Service(22, "tcp", "ssh"))

    elif device_type == "win_desktop":
        services.append(Service(3389, "tcp", "rdp"))
        if rng.random() < 0.3:
            services.append(Service(445, "tcp", "smb"))

    elif device_type == "iot_sensor":
        services.append(Service(1883, "tcp", "mqtt"))

    elif device_type == "db_server":
        services.append(Service(5432, "tcp", "postgres"))
        if rng.random() < 0.4:
            services.append(Service(22, "tcp", "ssh"))

    return services


# ---------------------------------------------------------
# Generate random response model
# ---------------------------------------------------------
def _random_response_model(is_honeypot: bool, rng: np.random.Generator) -> ResponseModel:
    """
    Honeypots have:
        - lower RTT variability
        - noisier banners
        - honeypot artifacts
    """
    rtt_mean = rng.uniform(5, 80)
    rtt_std = rng.uniform(1, 15)
    noise_base = rng.uniform(0.1, 0.5)

    banner = rng.choice(
        [
            "Apache/2.4.41 (Ubuntu)",
            "nginx/1.18.0",
            "OpenSSH_8.2p1",
            "Microsoft-IIS/10.0",
        ]
    )

    if is_honeypot:
        return ResponseModel(
            rtt_mean=rtt_mean,
            rtt_std=rtt_std * 0.3,              # honeypots respond very uniformly
            banner_template=banner,
            noise_level=noise_base * 0.5,
            honeypot_artefact_prob=rng.uniform(0.1, 0.4),
        )
    else:
        return ResponseModel(
            rtt_mean=rtt_mean,
            rtt_std=rtt_std,
            banner_template=banner,
            noise_level=noise_base,
            honeypot_artefact_prob=0.0,         # no artifacts for real hosts
        )


# ---------------------------------------------------------
# Public function: generate full host profile
# ---------------------------------------------------------
def generate_random_host(
    host_id: int,
    is_honeypot: bool,
    rng: np.random.Generator,
) -> Host:
    """
    Completely generate a host with services + response model.
    """
    device_type = rng.choice(DEVICE_TYPES)
    services = _random_services(device_type, rng)

    response_model = _random_response_model(is_honeypot, rng)
    host_type = HostType.HONEYPOT if is_honeypot else HostType.REAL

    return Host(
        host_id=host_id,
        host_type=host_type,
        device_type=device_type,
        services=services,
        response_model=response_model,
    )
