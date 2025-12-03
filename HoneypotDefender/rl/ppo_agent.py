from __future__ import annotations
import torch
import torch.nn as nn
from torch.distributions import Categorical


class GraphLayer(nn.Module):
    """
    Eenvoudige graph-convolution laag:
    - Voegt self-loops toe aan adjacency
    - Genormaliseerde buur-aggregatie
    - Combineert self + neighbor features

    h_in: (B, N, d)
    adj : (B, N, N) met 0/1
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.self_lin = nn.Linear(d_model, d_model)
        self.neigh_lin = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, adj):
        """
        h:   (B, N, d)
        adj: (B, N, N) met 0/1
        """
        B, N, d = h.shape
        device = h.device

        # self-loops
        eye = torch.eye(N, device=device).unsqueeze(0)  # (1, N, N)
        adj_with_self = adj + eye
        adj_with_self = adj_with_self.clamp(max=1.0)

        # normalisatie per rij
        deg = adj_with_self.sum(dim=-1, keepdim=True)  # (B, N, 1)
        deg = deg + 1e-6
        norm_adj = adj_with_self / deg  # (B, N, N)

        # neighbor aggregatie
        neigh = torch.bmm(norm_adj, h)  # (B, N, d)

        out = self.self_lin(h) + self.neigh_lin(neigh)
        out = self.act(out)
        out = self.norm(out)
        return out


class PPOAgent(nn.Module):
    """
    Graph-based PPO policy/value netwerk voor de defender.

    Werkt op de *flattened* observatie uit HoneypotDefenderEnv, maar
    pakt daarbinnen:

        - node_features:  (N_max, F)
        - adjacency:      (N_max, N_max)
        - remaining_budget: (1,)

    Node input = [features, remaining_budget_per_node]
    Daarna 2× GraphLayer, vervolgens per node een actie-head (K actions),
    en een value-head op het gemiddelde over alle nodes.
    """

    def __init__(self, obs_dim: int, n_actions: int, N_max: int, F_features: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.N_max = N_max
        self.F = F_features

        # Aantal actie-types per host (K zou 8 moeten zijn)
        assert n_actions % N_max == 0, "n_actions moet deelbaar zijn door N_max"
        self.K = n_actions // N_max

        self.d_model = 256

        # Node input: [node_features (F), remaining_budget (1)]
        self.input_dim = self.F + 1

        self.input_proj = nn.Linear(self.input_dim, self.d_model)

        # 2 graph-layers
        self.gnn_layers = nn.ModuleList(
            [GraphLayer(self.d_model), GraphLayer(self.d_model)]
        )

        # Policy: per host een vector van lengte K
        self.action_head = nn.Linear(self.d_model, self.K)

        # Value: gemiddelde over hosts → scalar
        self.value_head = nn.Linear(self.d_model, 1)

    # ---------------------------------------------------------
    # Helper: unpack flattened observation
    # ---------------------------------------------------------
    def _unpack_obs(self, x):
        """
        x: (B, obs_dim) = concat[
            node_features (N*F),
            adjacency (N*N),
            remaining_budget (1)
        ]
        """
        B, D = x.shape
        N = self.N_max
        F = self.F

        offset = 0

        # node_features
        nf_size = N * F
        nf_flat = x[:, offset : offset + nf_size]
        offset += nf_size
        node_features = nf_flat.view(B, N, F)

        # adjacency
        adj_size = N * N
        adj_flat = x[:, offset : offset + adj_size]
        offset += adj_size
        adjacency = adj_flat.view(B, N, N)

        # remaining_budget (scalar)
        remaining_budget = x[:, offset : offset + 1]  # (B, 1)

        return node_features, adjacency, remaining_budget

    # ---------------------------------------------------------
    # Forward: graph enc, policy-logits en value
    # ---------------------------------------------------------
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, obs_dim)

        B = x.shape[0]
        N = self.N_max

        node_features, adjacency, remaining_budget = self._unpack_obs(x)

        adjacency = adjacency.clamp(0.0, 1.0)

        # remaining_budget naar per-node feature
        budget = remaining_budget.view(B, 1, 1).expand(-1, N, 1)  # (B, N, 1)

        node_input = torch.cat([node_features, budget], dim=-1)  # (B, N, F+1)

        h = self.input_proj(node_input)  # (B, N, d_model)

        for layer in self.gnn_layers:
            h = layer(h, adjacency)

        # Policy head: per host K acties
        per_node_logits = self.action_head(h)  # (B, N, K)
        logits = per_node_logits.view(B, N * self.K)  # (B, n_actions)

        # Value head
        graph_emb = h.mean(dim=1)          # (B, d_model)
        value = self.value_head(graph_emb) # (B, 1)
        value = value.squeeze(-1)          # (B,)

        return logits, value

    # ---------------------------------------------------------
    # API voor PPO: actie + logprob + entropy + value
    # ---------------------------------------------------------
    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy, value
