import torch
import torch.nn as nn
from torch.distributions import Categorical

class GraphLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_lin = nn.Linear(d_model, d_model)
        self.neigh_lin = nn.Linear(d_model, d_model)
        self.act = nn.ReLU()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, adj):
        _, N, _ = h.shape

        device = h.device
        eye = torch.eye(N, device=device).unsqueeze(0)
        adj_with_self = adj + eye
        adj_with_self = adj_with_self.clamp(max=1.0)

        deg = adj_with_self.sum(dim=-1, keepdim=True)
        deg = deg + 1e-6
        norm_adj = adj_with_self / deg

        neigh = torch.bmm(norm_adj, h)

        out = self.self_lin(h) + self.neigh_lin(neigh)
        out = self.act(out)
        out = self.norm(out)
        return out

class PPOAgent(nn.Module):
    def __init__(self, obs_dim, n_actions, N_max, F_features):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.N_max = N_max
        self.F = F_features

        self.K = n_actions // N_max

        self.d_model = 256

        self.input_dim = self.F + 3

        self.input_proj = nn.Linear(self.input_dim, self.d_model)

        self.gnn_layers = nn.ModuleList([GraphLayer(self.d_model), GraphLayer(self.d_model)])

        self.action_head = nn.Linear(self.d_model, self.K)

        self.value_head = nn.Linear(self.d_model, 1)

    def _unpack_obs(self, x):
        B, _ = x.shape
        N = self.N_max
        F = self.F

        offset = 0

        nf_size = N * F
        nf_flat = x[:, offset : offset + nf_size]
        offset += nf_size
        node_features = nf_flat.view(B, N, F)

        adj_size = N * N
        adj_flat = x[:, offset : offset + adj_size]
        offset += adj_size
        adjacency = adj_flat.view(B, N, N)

        mk_size = N
        mk_flat = x[:, offset : offset + mk_size]
        offset += mk_size
        mask_known = mk_flat.view(B, N)

        mc_size = N
        mc_flat = x[:, offset : offset + mc_size]
        offset += mc_size
        mask_classified = mc_flat.view(B, N)

        remaining_budget = x[:, offset : offset + 1]

        return node_features, adjacency, mask_known, mask_classified, remaining_budget

    def forward(self, x, action_mask=None):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        B = x.shape[0]
        N = self.N_max
        K = self.K

        node_features, adjacency, mask_known, mask_classified, remaining_budget = (self._unpack_obs(x))

        adjacency = adjacency.clamp(0.0, 1.0)

        known = mask_known.unsqueeze(-1)
        classified = mask_classified.unsqueeze(-1)
        budget = remaining_budget.view(B, 1, 1).expand(-1, N, 1)

        node_input = torch.cat([node_features, known, classified, budget], dim=-1)

        h = self.input_proj(node_input)

        for layer in self.gnn_layers:
            h = layer(h, adjacency)

        per_node_logits = self.action_head(h)

        logits = per_node_logits.view(B, N * K)

        graph_emb = h.mean(dim=1)
        value = self.value_head(graph_emb)
        value = value.squeeze(-1)

        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                mask = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
            else:
                mask = action_mask.to(dtype=torch.bool, device=logits.device)

            if mask.dim() == 1:
                mask = mask.unsqueeze(0)

            if mask.shape != logits.shape:
                if mask.shape[0] == 1 and logits.shape[0] > 1:
                    mask = mask.expand_as(logits)
                else:
                    raise ValueError(f"action_mask shape {mask.shape} does not match logits {logits.shape}")

            logits = logits.masked_fill(~mask, -1e9)

        return logits, value

    def get_action_and_value(self, x, action=None, action_mask=None):
        logits, value = self(x, action_mask=action_mask)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy, value
