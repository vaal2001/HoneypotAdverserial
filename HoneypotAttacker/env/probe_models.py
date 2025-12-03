from dataclasses import dataclass
from typing import Optional
import numpy as np

from .host_profiles import Host, HostType, Service


@dataclass
class ProbeResult:
    """
    Unified result for all probe actions.
    """
    success: bool
    rtt: float

    # For SYN scan
    port_open: Optional[bool] = None

    # For banner & OS probes
    banner_score: float = 0.0
    os_guess_score: float = 0.0

    # Whether the response showed honeypot artefacts
    honeypot_artefact_flag: bool = False


# -------------------------------------------------------
# Internal helper
# -------------------------------------------------------

def _find_service(host: Host, port: int) -> Optional[Service]:
    """Find a service running on 'port', else None."""
    for s in host.services:
        if s.port == port:
            return s
    return None


# -------------------------------------------------------
# Probe Simulations
# -------------------------------------------------------

def simulate_ping(host: Host, rng: np.random.Generator) -> ProbeResult:
    """
    Ping is always successful; provides noisy RTT estimate.
    """
    rtt = max(1.0, rng.normal(host.response_model.rtt_mean,
                              host.response_model.rtt_std))

    return ProbeResult(
        success=True,
        rtt=rtt,
    )


def simulate_syn_scan(host: Host, port: int, rng: np.random.Generator) -> ProbeResult:
    """
    SYN scan reveals whether port is open, plus potential honeypot artefact.
    """
    service = _find_service(host, port)

    port_open = service is not None
    rtt = max(1.0, rng.normal(host.response_model.rtt_mean,
                              host.response_model.rtt_std))

    return ProbeResult(
        success=True,
        rtt=rtt,
        port_open=port_open,
        honeypot_artefact_flag=(
            host.host_type == HostType.HONEYPOT
            and rng.random() < host.response_model.honeypot_artefact_prob
        ),
    )


def simulate_banner_grab(host: Host, rng: np.random.Generator) -> ProbeResult:
    """
    Banner-grab reveals software banner information with noise.
    Honeypots typically produce different-looking banners.
    """
    rtt = max(1.0, rng.normal(host.response_model.rtt_mean,
                              host.response_model.rtt_std))

    # REAL ~ 1.0, HONEYPOT ~ 2.0 (but noisy)
    base = 1.0 if host.host_type == HostType.REAL else 2.0
    banner = rng.normal(base, 0.3)

    return ProbeResult(
        success=True,
        rtt=rtt,
        banner_score=banner,
        honeypot_artefact_flag=(
            host.host_type == HostType.HONEYPOT
            and rng.random() < host.response_model.honeypot_artefact_prob
        ),
    )


def simulate_os_probe(host: Host, rng: np.random.Generator) -> ProbeResult:
    """
    OS fingerprint + RTT.
    Real: ~1.0 score
    Honeypot: ~0.7 score
    """
    rtt = max(1.0, rng.normal(host.response_model.rtt_mean,
                              host.response_model.rtt_std))

    base = 1.0 if host.host_type == HostType.REAL else 0.7
    os_guess = rng.normal(base, 0.2)

    return ProbeResult(
        success=True,
        rtt=rtt,
        os_guess_score=os_guess,
    )


def simulate_service_probe(host: Host, rng: np.random.Generator) -> ProbeResult:
    """
    This probe may produce honeypot artefacts if interacting with fake services.
    """
    rtt = max(1.0, rng.normal(host.response_model.rtt_mean,
                              host.response_model.rtt_std))

    return ProbeResult(
        success=True,
        rtt=rtt,
        honeypot_artefact_flag=(
            host.host_type == HostType.HONEYPOT
            and rng.random() < host.response_model.honeypot_artefact_prob
        ),
    )
