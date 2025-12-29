from dataclasses import dataclass
from typing import Optional

from .host_profiles import HostType

@dataclass
class ProbeResult:
    success: bool
    rtt: float
    port_open: Optional[bool] = None
    banner_score: float = 0.0
    os_guess_score: float = 0.0
    honeypot_artefact_flag: bool = False

def _find_service(host, port):
    for s in host.services:
        if s.port == port:
            return s
    return None

def simulate_ping(host, rng):
    rtt = max(1.0, rng.normal(host.response_model.rtt_mean, host.response_model.rtt_std))

    return ProbeResult(success=True, rtt=rtt)

def simulate_syn_scan(host, port, rng):
    service = _find_service(host, port)

    port_open = service is not None
    rtt = max(1.0, rng.normal(host.response_model.rtt_mean, host.response_model.rtt_std))

    return ProbeResult(
        success=True,
        rtt=rtt,
        port_open=port_open,
        honeypot_artefact_flag=(host.host_type == HostType.HONEYPOT and rng.random() < host.response_model.honeypot_artefact_prob)
    )

def simulate_banner_grab(host, rng):
    rtt = max(1.0, rng.normal(host.response_model.rtt_mean, host.response_model.rtt_std))
    base = 1.0 if host.host_type == HostType.REAL else 2.0
    banner = rng.normal(base, 0.3)

    return ProbeResult(
        success=True,
        rtt=rtt,
        banner_score=banner,
        honeypot_artefact_flag=(host.host_type == HostType.HONEYPOT and rng.random() < host.response_model.honeypot_artefact_prob)
    )

def simulate_os_probe(host, rng):
    rtt = max(1.0, rng.normal(host.response_model.rtt_mean, host.response_model.rtt_std))

    base = 1.0 if host.host_type == HostType.REAL else 0.7
    os_guess = rng.normal(base, 0.2)

    return ProbeResult(success=True, rtt=rtt, os_guess_score=os_guess)

def simulate_service_probe(host, rng):
    rtt = max(1.0, rng.normal(host.response_model.rtt_mean, host.response_model.rtt_std))

    return ProbeResult(
        success=True,
        rtt=rtt,
        honeypot_artefact_flag=(host.host_type == HostType.HONEYPOT and rng.random() < host.response_model.honeypot_artefact_prob),
    )
