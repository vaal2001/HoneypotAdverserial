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

        # self-loops toevoegen
        device = h.device
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
    Graph-based PPO policy/value netwerk.

    Werkt nog steeds op de *flattened* observatie zoals je env nu bouwt,
    maar pakt daarbinnen de node_features + adjacency + masks uit en
    bouwt een graph-embedding per host.

    - Node input: [features, known_flag, classified_flag, remaining_budget]
    - 2× GraphLayer (d_model = 256)
    - Per host een action-head → (B, N_max, K)
    - Flatten naar (B, N_max*K) zodat bestaande PPO-loop gewoon blijft werken
    """

    def __init__(self, obs_dim, n_actions, N_max, F_features):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.N_max = N_max
        self.F = F_features

        # Aantal actie-types per host (zou 8 moeten zijn)
        self.K = n_actions // N_max

        # Graph encoder parameters
        self.d_model = 256

        # We plakken known, classified en remaining_budget per node erbij:
        # [node_features (F), known (1), classified (1), budget (1)]
        self.input_dim = self.F + 3

        self.input_proj = nn.Linear(self.input_dim, self.d_model)

        # 2 graph-layers (zoals afgesproken)
        self.gnn_layers = nn.ModuleList(
            [GraphLayer(self.d_model), GraphLayer(self.d_model)]
        )

        # Policy: per host een actie-vector van lengte K
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
            mask_known (N),
            mask_classified (N),
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

        # mask_known
        mk_size = N
        mk_flat = x[:, offset : offset + mk_size]
        offset += mk_size
        mask_known = mk_flat.view(B, N)

        # mask_classified
        mc_size = N
        mc_flat = x[:, offset : offset + mc_size]
        offset += mc_size
        mask_classified = mc_flat.view(B, N)

        # remaining_budget (scalar)
        remaining_budget = x[:, offset : offset + 1]  # (B, 1)

        return node_features, adjacency, mask_known, mask_classified, remaining_budget

    # ---------------------------------------------------------
    # Forward: bouw graph, policy-logits en value
    # ---------------------------------------------------------
    def forward(self, x, action_mask=None):
        """
        x: (B, obs_dim) flattened observation
        action_mask: (B, n_actions) of (n_actions,) boolean; True = allowed
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, obs_dim)

        B = x.shape[0]
        N = self.N_max
        K = self.K

        node_features, adjacency, mask_known, mask_classified, remaining_budget = (
            self._unpack_obs(x)
        )

        # Zorg dat adjacency in [0,1] zit en float
        adjacency = adjacency.clamp(0.0, 1.0)

        # Maak extra per-node features
        known = mask_known.unsqueeze(-1)          # (B, N, 1)
        classified = mask_classified.unsqueeze(-1)  # (B, N, 1)
        budget = remaining_budget.view(B, 1, 1).expand(-1, N, 1)  # (B, N, 1)

        # Combineer tot node input
        node_input = torch.cat(
            [node_features, known, classified, budget], dim=-1
        )  # (B, N, F+3)

        h = self.input_proj(node_input)  # (B, N, d_model)

        # 2 graph-layers
        for layer in self.gnn_layers:
            h = layer(h, adjacency)

        # Policy head: per host K acties
        # h: (B, N, d_model) → (B, N, K)
        per_node_logits = self.action_head(h)

        # Flatten naar (B, N*K) zodat je PPO-loop hetzelfde blijft
        logits = per_node_logits.view(B, N * K)

        # Value head: gemiddelde over hosts
        graph_emb = h.mean(dim=1)          # (B, d_model)
        value = self.value_head(graph_emb) # (B, 1)
        value = value.squeeze(-1)          # (B,)

        # HARD MASK toepassen (zoals je oude versie)
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                mask = torch.tensor(
                    action_mask, dtype=torch.bool, device=logits.device
                )
            else:
                mask = action_mask.to(dtype=torch.bool, device=logits.device)

            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # (1, n_actions)

            if mask.shape != logits.shape:
                if mask.shape[0] == 1 and logits.shape[0] > 1:
                    mask = mask.expand_as(logits)
                else:
                    raise ValueError(
                        f"action_mask shape {mask.shape} does not match logits {logits.shape}"
                    )

            # Illegal actions krijgen een gigantische negatieve logit
            logits = logits.masked_fill(~mask, -1e9)

        return logits, value

    # ---------------------------------------------------------
    # API voor PPO: actie + logprob + entropy + value
    # ---------------------------------------------------------
    def get_action_and_value(self, x, action=None, action_mask=None):
        """
        Gebruikt tijdens rollouts:
        - bouwt masked logits
        - sampled actie
        - geeft logprob, entropy en value terug
        """
        logits, value = self(x, action_mask=action_mask)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy, value
