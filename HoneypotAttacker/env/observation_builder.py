import numpy as np
from typing import Dict, Any


def build_observation(
    features: np.ndarray,
    adj_disc: np.ndarray,
    mask_known: np.ndarray,
    mask_classified: np.ndarray,
    remaining: int,
    max_budget: int,
) -> Dict[str, Any]:
    """
    Builds a complete observation dictionary for HoneypotDetectionEnv.

    Parameters
    ----------
    features : np.ndarray
        Node feature matrix with shape (N_max, F)
    adj_disc : np.ndarray
        Discovered adjacency matrix (N_max, N_max)
    mask_known : np.ndarray
        Binary mask (N_max,)
    mask_classified : np.ndarray
        Binary mask (N_max,)
    remaining : int
        Remaining action budget
    max_budget : int
        Initial max action budget to normalize remaining

    Returns
    -------
    dict
        {
            "node_features": (N_max, F),
            "adjacency": (N_max, N_max),
            "mask_known": (N_max,),
            "mask_classified": (N_max,),
            "remaining_budget": (1,)
        }
    """

    # Normalized remaining budget
    remaining_norm = np.array([remaining / max_budget], dtype=np.float32)

    return {
        "node_features": features.astype(np.float32),
        "adjacency": adj_disc.astype(np.float32),
        "mask_known": mask_known.astype(np.int8),
        "mask_classified": mask_classified.astype(np.int8),
        "remaining_budget": remaining_norm,
    }
