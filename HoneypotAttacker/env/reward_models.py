from .host_profiles import HostType

def classification_reward(host, classified_as_honeypot):
    if host.host_type == HostType.REAL and not classified_as_honeypot:
        return +4
    if host.host_type == HostType.HONEYPOT and classified_as_honeypot:
        return +4

    if host.host_type == HostType.REAL and classified_as_honeypot:
        return -6

    if host.host_type == HostType.HONEYPOT and not classified_as_honeypot:
        return -3
    return -4

def step_cost():
    return -0.01

def honeypot_trigger_penalty():
    return -3.0

def budget_exhausted_penalty(unclassified, total):
    if total == 0:
        return 0.0

    return -2.0 * float(unclassified)

def probe_reward():
    return 0.2

def intrinsic_probe_signal(host, result):
    return 0.1 if getattr(result, "honeypot_artefact_flag", False) else 0.0

def early_finish_bonus(remaining_actions, max_actions):
    return 1.0 * (remaining_actions / max_actions)

def probe_reward_for_type(probe_type):
    if probe_type == 1:  # SYN
        return 0.05

    if probe_type == 2:  # BANNER
        return 0.20

    if probe_type == 3:  # OS
        return 0.20

    if probe_type == 4:  # SERVICE
        return 0.25

    return 0.0

