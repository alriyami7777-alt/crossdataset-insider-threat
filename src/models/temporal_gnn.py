"""
Temporal heterogeneous GNN backbone (TGN-style memory + heterogeneous message
passing). This is Contribution 2's detector. Requires torch + torch_geometric.

Design:
  * Each node (user/host) keeps a memory vector updated as timestamped access
    edges stream in (TGN: message -> aggregate -> GRU memory update).
  * A time-encoding maps delta-t into the message so recency is explicit.
  * Heterogeneous relation-specific projections handle user vs host.
  * An edge-scoring head outputs an anomaly score per access event.

Kept deliberately compact and readable; Cursor should extend (multi-layer HGT,
neighbour sampling for scale) against the real graphs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """Learnable Fourier features of delta-t (Time2Vec-style)."""
    def __init__(self, dim=16):
        super().__init__()
        self.w = nn.Linear(1, dim)

    def forward(self, dt):                       # dt: [E,1]
        return torch.cos(self.w(dt))


class MemoryModule(nn.Module):
    def __init__(self, n_nodes, mem_dim=64):
        super().__init__()
        self.mem_dim = mem_dim
        self.register_buffer("memory", torch.zeros(n_nodes, mem_dim))
        self.register_buffer("last_t", torch.zeros(n_nodes))
        self.gru = nn.GRUCell(mem_dim, mem_dim)

    def reset(self):
        self.memory.zero_()
        self.last_t.zero_()

    def resize(self, n_nodes):
        """Reallocate memory for a different-sized graph (cross-dataset scoring)
        WITHOUT touching learned weights -- this is what makes the model inductive."""
        dev = self.memory.device
        self.memory = torch.zeros(n_nodes, self.mem_dim, device=dev)
        self.last_t = torch.zeros(n_nodes, device=dev)

    def update(self, node_ids, messages):
        prev = self.memory[node_ids]
        new = self.gru(messages, prev)
        self.memory = self.memory.clone()
        self.memory[node_ids] = new.detach()     # detach: TGN-style stop-grad
        return new


class TemporalHeteroGNN(nn.Module):
    def __init__(self, n_users, n_hosts, n_actions=16, mem_dim=64, time_dim=16,
                 feat_dim=0, use_time_encoding: bool = True):
        super().__init__()
        self.mem_dim = mem_dim
        self.feat_dim = feat_dim
        self.time_dim = time_dim
        self.use_time_encoding = bool(use_time_encoding)
        self.user_mem = MemoryModule(n_users, mem_dim)
        self.host_mem = MemoryModule(n_hosts, mem_dim)
        self.time_enc = TimeEncoder(time_dim)
        self.action_emb = nn.Embedding(n_actions, 16)
        msg_in = mem_dim + 16 + time_dim
        self.user_msg = nn.Linear(msg_in, mem_dim)
        self.host_msg = nn.Linear(msg_in, mem_dim)
        self.scorer = nn.Sequential(
            nn.Linear(2 * mem_dim, mem_dim), nn.ReLU(),
            nn.Linear(mem_dim, 1),
        )
        # self-supervised head: predict an edge's action from node memories
        self.action_head = nn.Linear(2 * mem_dim, n_actions)
        # User-day readout: r_{u,d} = [s_u || x_{u,d}]  (paper Eq.)
        ud_in = mem_dim + max(feat_dim, 0)
        self.userday_head = nn.Sequential(
            nn.Linear(ud_in, mem_dim), nn.ReLU(),
            nn.Linear(mem_dim, 1),
        )

    def score_user_day_vec(self, user_mem, x_ud):
        """Logit from concatenated user memory and user-day features."""
        if x_ud is None or self.feat_dim <= 0:
            # fallback: memory-only (zero features)
            x_ud = user_mem.new_zeros(user_mem.shape[0], 0)
        return self.userday_head(torch.cat([user_mem, x_ud], dim=-1)).squeeze(-1)

    def reset_memory(self):
        self.user_mem.reset()
        self.host_mem.reset()

    def set_graph(self, n_users, n_hosts):
        """Point the memory at a new dataset's node counts; learned weights kept."""
        self.user_mem.resize(n_users)
        self.host_mem.resize(n_hosts)

    def predict_action(self, u, h):
        """Pre-update action prediction (for self-supervised pretraining)."""
        emb = torch.cat([self.user_mem.memory[u], self.host_mem.memory[h]], dim=-1)
        return self.action_head(emb)

    def forward(self, u, h, a, dt):
        """u,h: node ids [E]; a: action ids [E]; dt: delta-t [E,1].
        Returns per-edge anomaly logit [E] and the fused edge embedding."""
        if self.use_time_encoding:
            te = self.time_enc(dt)                # [E,time_dim]
        else:
            te = dt.new_zeros(dt.size(0), self.time_dim)
        ae = self.action_emb(a)                   # [E,16]
        um, hm = self.user_mem.memory[u], self.host_mem.memory[h]
        u_in = torch.cat([um, ae, te], dim=-1)
        h_in = torch.cat([hm, ae, te], dim=-1)
        u_new = self.user_mem.update(u, self.user_msg(u_in))
        h_new = self.host_mem.update(h, self.host_msg(h_in))
        edge_emb = torch.cat([u_new, h_new], dim=-1)
        logit = self.scorer(edge_emb).squeeze(-1)
        return logit, edge_emb


def self_supervised_loss(model, u, h, a, dt):
    """Label-free next-action prediction: predict each edge's action from the
    CURRENT node memories (before the event is seen), then fold the event into
    memory. Forces memories to encode behavioural regularities pre-labels."""
    import torch.nn.functional as F
    logits = model.predict_action(u, h)      # pre-update
    loss = F.cross_entropy(logits, a)
    model(u, h, a, dt)                        # update memory with the observed event
    return loss
