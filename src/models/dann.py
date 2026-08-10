"""
Domain-adversarial training (DANN) wrapper -- the mechanism that closes the
cross-dataset gap in Contribution 2.

A gradient-reversal layer sits between the shared encoder and a domain classifier.
The encoder is trained to (a) detect insiders on the source AND (b) FOOL the
domain classifier, so its features become dataset-invariant and transfer to the
target. Verify the gradient-reversal sign convention and the lambda schedule with
ChatGPT (see collaborator-prompts.md, ChatGPT item 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    return _GradReverse.apply(x, lambd)


class DomainAdversary(nn.Module):
    """Domain classifier applied to the (gradient-reversed) edge embedding."""
    def __init__(self, emb_dim, n_domains=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, emb, lambd=1.0):
        return self.net(grad_reverse(emb, lambd))


def lambda_schedule(step, total_steps, gamma=10.0):
    """Standard DANN annealing: 2/(1+exp(-gamma*p)) - 1, p in [0,1]."""
    p = min(1.0, step / max(1, total_steps))
    return 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p))).item() - 1.0
