from .shared_host_profiles import HostType
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProbeResult:
    success: bool
    rtt: float
    port_open: Optional[bool] = None
    banner_score: float = 0.0
    os_guess_score: float = 0.0
    artefact: bool = False

def simulate_ping(host, rng):
    rm = host.response_model
    rtt = max(1.0, rng.normal(rm.rtt_mean, rm.rtt_std))
    return ProbeResult(True, rtt)

def simulate_syn_scan(host, rng):
    rm = host.response_model
    rtt = max(1.0, rng.normal(rm.rtt_mean, rm.rtt_std))
    open_port = rm.service_count >= 1
    artefact = (host.host_type == HostType.HONEYPOT and rng.random() < rm.artefact_prob)
    return ProbeResult(True, rtt, port_open=open_port, artefact=artefact)

def simulate_banner(host, rng):
    rm = host.response_model
    rtt = max(1.0, rng.normal(rm.rtt_mean, rm.rtt_std))
    base = 1.0 if host.host_type == HostType.REAL else 2.0
    banner = rng.normal(base, rm.banner_noise)
    artefact = (host.host_type == HostType.HONEYPOT and rng.random() < rm.artefact_prob)
    return ProbeResult(True, rtt, banner_score=banner, artefact=artefact)

def simulate_os(host, rng):
    rm = host.response_model
    rtt = max(1.0, rng.normal(rm.rtt_mean, rm.rtt_std))
    guess = rng.normal(rm.os_fingerprint, 0.1)
    return ProbeResult(True, rtt, os_guess_score=guess)

def simulate_service(host, rng):
    rm = host.response_model
    rtt = max(1.0, rng.normal(rm.rtt_mean, rm.rtt_std))
    artefact = (host.host_type == HostType.HONEYPOT and rng.random() < rm.artefact_prob)
    return ProbeResult(True, rtt, artefact=artefact)
