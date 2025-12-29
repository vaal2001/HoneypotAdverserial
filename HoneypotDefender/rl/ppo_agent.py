import torch
import torch.nn as nn
from torch.distributions import Categorical

class GraphLayer(nn.Module):
    """
    Simple graph convolution layer:
    """

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
    """
    Graph-based PPO policy/value network for the defender.
    """

    def __init__(self, obs_dim, n_actions, N_max, F_features):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.N_max = N_max
        self.F = F_features

        assert n_actions % N_max == 0, "n_actions moet deelbaar zijn door N_max"
        self.K = n_actions // N_max

        self.d_model = 256

        self.input_dim = self.F + 1

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

        remaining_budget = x[:, offset : offset + 1]

        return node_features, adjacency, remaining_budget

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        B = x.shape[0]
        N = self.N_max

        node_features, adjacency, remaining_budget = self._unpack_obs(x)

        adjacency = adjacency.clamp(0.0, 1.0)

        budget = remaining_budget.view(B, 1, 1).expand(-1, N, 1)

        node_input = torch.cat([node_features, budget], dim=-1)

        h = self.input_proj(node_input)

        for layer in self.gnn_layers:
            h = layer(h, adjacency)

        per_node_logits = self.action_head(h)
        logits = per_node_logits.view(B, N * self.K)

        graph_emb = h.mean(dim=1)
        value = self.value_head(graph_emb)
        value = value.squeeze(-1)

        return logits, value

    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy, value
