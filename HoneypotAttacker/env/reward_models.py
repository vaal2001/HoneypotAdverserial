from .host_profiles import Host, HostType


def classification_reward(host: Host, classified_as_honeypot: bool) -> float:
    """
    Asymmetrische cost:
      - Goed is altijd +4
      - REAL → HONEYPOT fout (false positive) is -6
      - HONEYPOT → REAL fout (false negative) is -3
    """
    if host.host_type == HostType.REAL and not classified_as_honeypot:
        return +4  # correcte real
    if host.host_type == HostType.HONEYPOT and classified_as_honeypot:
        return +4  # correcte honeypot

    # Fouten:
    if host.host_type == HostType.REAL and classified_as_honeypot:
        return -6  # false positive, zwaarder bestraft

    if host.host_type == HostType.HONEYPOT and not classified_as_honeypot:
        return -3  # false negative, iets minder zwaar

    return -4


def step_cost() -> float:
    return -0.01


def honeypot_trigger_penalty() -> float:
    """
    Mildere straf voor het triggeren van een honeypot artefact via service probe.
    Geen episode-stop meer.
    """
    return -3.0


def budget_exhausted_penalty(unclassified: int, total: int) -> float:
    """
    Penalty when the action budget is exhausted.
    Scales with number of unclassified hosts.
    """
    if total == 0:
        return 0.0

    # e.g. -2 per unclassified host
    return -2.0 * float(unclassified)


def probe_reward() -> float:
    """
    Sterkere stimulans voor informatie vergaren.
    """
    return 0.2


def intrinsic_probe_signal(host, result) -> float:
    """
    Artefact bonus blijft maar kleiner.
    """
    return 0.1 if getattr(result, "honeypot_artefact_flag", False) else 0.0


def early_finish_bonus(remaining_actions, max_actions):
    return 1.0 * (remaining_actions / max_actions)

def probe_reward_for_type(probe_type: int) -> float:
    """
    Differentieel probe reward systeem.

    Probe types:
      1 = SYN_SCAN
      2 = BANNER_GRAB   (zeer informatief → hogere reward)
      3 = OS_PROBE      (zeer informatief → hogere reward)
      4 = SERVICE_PROBE (informatief, maar met risico → medium reward)
    """
    if probe_type == 1:  # SYN
        return 0.05

    if probe_type == 2:  # BANNER
        return 0.20

    if probe_type == 3:  # OS
        return 0.20

    if probe_type == 4:  # SERVICE
        return 0.25

    return 0.0

