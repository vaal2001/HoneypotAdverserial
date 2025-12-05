# attacker_rewards.py
from .shared_host_profiles import HostType

def attacker_reward(host, classified_as_honeypot: bool) -> float:
    if host.host_type == HostType.REAL and not classified_as_honeypot:
        return +4
    if host.host_type == HostType.HONEYPOT and classified_as_honeypot:
        return +4
    if host.host_type == HostType.REAL and classified_as_honeypot:
        return -6
    if host.host_type == HostType.HONEYPOT and not classified_as_honeypot:
        return -3
    return -4

def step_cost() -> float:
    return -0.01
