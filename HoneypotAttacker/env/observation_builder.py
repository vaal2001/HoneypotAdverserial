import numpy as np

def build_observation(features, adj_disc, mask_known, mask_classified, remaining, max_budget):
    remaining_norm = np.array([remaining / max_budget], dtype=np.float32)

    return {"node_features": features.astype(np.float32), "adjacency": adj_disc.astype(np.float32), "mask_known": mask_known.astype(np.int8), "mask_classified": mask_classified.astype(np.int8), "remaining_budget": remaining_norm}