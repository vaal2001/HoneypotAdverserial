from dataclasses import dataclass
from enum import Enum, auto
from typing import List

class HostType(Enum):
    REAL = auto()
    HONEYPOT = auto()

@dataclass
class Service:
    port: int
    protocol: str
    service_type: str

@dataclass
class ResponseModel:
    rtt_mean: float
    rtt_std: float
    banner_template: str
    noise_level: float
    honeypot_artefact_prob: float

@dataclass
class Host:
    host_id: int
    host_type: HostType
    device_type: str
    services: List[Service]
    response_model: ResponseModel

DEVICE_TYPES = [
    "linux_web",
    "win_desktop",
    "iot_sensor",
    "db_server",
]

def _random_services(device_type, rng):
    services = []

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

def _random_response_model(is_honeypot, rng):
    rtt_mean = rng.uniform(5, 80)
    rtt_std = rng.uniform(1, 15)
    noise_base = rng.uniform(0.1, 0.5)

    banner = rng.choice(["Apache/2.4.41 (Ubuntu)", "nginx/1.18.0", "OpenSSH_8.2p1", "Microsoft-IIS/10.0"])

    if is_honeypot:
        return ResponseModel(rtt_mean, rtt_std * 0.3, banner, noise_base * 0.5, rng.uniform(0.1, 0.4))
    else:
        return ResponseModel(rtt_mean, rtt_std, banner, noise_base, 0.0)

def generate_random_host(host_id, is_honeypot, rng):
    device_type = rng.choice(DEVICE_TYPES)
    services = _random_services(device_type, rng)

    response_model = _random_response_model(is_honeypot, rng)
    host_type = HostType.HONEYPOT if is_honeypot else HostType.REAL

    return Host(host_id, host_type, device_type, services, response_model)
