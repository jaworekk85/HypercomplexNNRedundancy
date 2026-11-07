"""Deterministic Sinkhorn channel assignment and export to a fixed permutation.

The relaxation is an existing mechanism (Mena2018), not a new method or a
full reproduction of stochastic Gumbel-Sinkhorn training. Hard assignment
uses the standard primal-dual assignment algorithm, independently tested on
exhaustively solved small cases. No SciPy dependency is required.
"""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.models.cifar_cnn_torch import CifarCNN


def maximum_assignment(scores):
    """Return one column per row maximizing a finite square score matrix."""
    cost = -np.asarray(scores, dtype=np.float64)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1] or not len(cost) or not np.isfinite(cost).all():
        raise ValueError("Expected a nonempty finite square assignment matrix")
    n = len(cost)
    u, v = np.zeros(n+1), np.zeros(n+1)
    p, way = np.zeros(n+1, dtype=np.int64), np.zeros(n+1, dtype=np.int64)
    for row in range(1, n+1):
        p[0], column = row, 0
        minimum, used = np.full(n+1, np.inf), np.zeros(n+1, dtype=bool)
        while True:
            used[column] = True
            active_row, delta, next_column = p[column], np.inf, 0
            for candidate in range(1, n+1):
                if not used[candidate]:
                    reduced = cost[active_row-1, candidate-1] - u[active_row] - v[candidate]
                    if reduced < minimum[candidate]:
                        minimum[candidate], way[candidate] = reduced, column
                    if minimum[candidate] < delta:
                        delta, next_column = minimum[candidate], candidate
            for candidate in range(n+1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    indices = np.empty(n, dtype=np.int64)
    for column in range(1, n+1):
        indices[p[column]-1] = column-1
    return indices


def log_sinkhorn(logits, temperature=1.0, iterations=30):
    """Differentiable square relaxation; positive finite temperature required."""
    values = logits / temperature
    if logits.ndim != 2 or logits.shape[0] != logits.shape[1] or iterations < 1:
        raise ValueError("Expected a square matrix and positive iteration count")
    for _ in range(iterations):
        values = values - torch.logsumexp(values, dim=1, keepdim=True)
        values = values - torch.logsumexp(values, dim=0, keepdim=True)
    return values.exp()


def _restore_routing_gradient_state(module, incompatible_keys):
    # Parameter.requires_grad is not serialized in state_dict; restore the phase.
    module.logits.requires_grad_(not bool(module.hard_enabled))


class FixedChannelPermutation(nn.Module):
    def __init__(self, indices):
        super().__init__()
        indices = torch.as_tensor(indices, dtype=torch.int64).detach().clone()
        if indices.ndim != 1 or not len(indices) or len(indices) % 4 or not torch.equal(indices.sort().values, torch.arange(len(indices), device=indices.device)):
            raise ValueError("Expected a permutation of a multiple of four channels")
        self.register_buffer("indices", indices)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != len(self.indices):
            raise ValueError("Unexpected channel routing shape")
        return inputs.index_select(1, self.indices)


class LearnableChannelPermutation(nn.Module):
    def __init__(self, channels, seed, initial_strength=4.0, temperature=1.0, iterations=30):
        super().__init__()
        if channels < 4 or channels % 4 or temperature <= 0 or iterations < 1:
            raise ValueError("Invalid channel-assignment configuration")
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(channels, generator=generator)
        logits = .01*torch.randn(channels, channels, generator=generator)
        logits[torch.arange(channels), indices] += initial_strength
        self.logits = nn.Parameter(logits)
        self.register_buffer("initial_indices", indices.clone())
        self.register_buffer("indices", indices.clone())
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("hard_enabled", torch.tensor(False))
        self.iterations = iterations
        self.register_load_state_dict_post_hook(_restore_routing_gradient_state)

    def soft_matrix(self):
        return log_sinkhorn(self.logits, self.temperature, self.iterations)

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != len(self.indices):
            raise ValueError("Unexpected channel routing shape")
        if bool(self.hard_enabled):
            return inputs.index_select(1, self.indices)
        return torch.einsum("ij,njhw->nihw", self.soft_matrix().to(inputs.dtype), inputs)

    @torch.no_grad()
    def harden(self):
        indices = maximum_assignment(self.logits.detach().cpu().double().numpy())
        self.indices.copy_(torch.as_tensor(indices, device=self.indices.device))
        self.hard_enabled.fill_(True)
        self.logits.requires_grad_(False)

    @torch.no_grad()
    def export(self):
        if not bool(self.hard_enabled):
            raise ValueError("Harden before exporting")
        return FixedChannelPermutation(self.indices)

    @torch.no_grad()
    def diagnostics(self):
        matrix = self.soft_matrix().double()
        chosen = maximum_assignment(self.logits.detach().cpu().double().numpy())
        return {"row_sum_max_error": float((matrix.sum(1)-1).abs().max()),
                "column_sum_max_error": float((matrix.sum(0)-1).abs().max()),
                "mean_row_entropy": float(-(matrix*matrix.clamp_min(1e-30).log()).sum()/len(matrix)),
                "matching": chosen.tolist(), "initial_matching": self.initial_indices.cpu().tolist(),
                "changed_output_positions": int(np.sum(chosen != self.initial_indices.cpu().numpy())),
                "hard_enabled": bool(self.hard_enabled), "temperature": float(self.temperature)}


class RoutedCifarCNN(CifarCNN):
    def __init__(self, mode, seed=2026, q_widths=(8, 16, 32), dropout=(.1, .2, .3),
                 initial_strength=4.0, temperature=1.0, iterations=30):
        super().__init__("quaternion", q_widths=q_widths, dropout=dropout)
        channels = 4*q_widths[0]
        if mode == "identity":
            self.router = FixedChannelPermutation(torch.arange(channels))
        elif mode == "random":
            self.router = FixedChannelPermutation(torch.randperm(channels, generator=torch.Generator().manual_seed(seed+60000)))
        elif mode == "learned":
            self.router = LearnableChannelPermutation(channels, seed+60000, initial_strength, temperature, iterations)
        else:
            raise ValueError("Unknown routing mode")
        self.routing_mode = mode

    def forward_features(self, rgb):
        values = torch.cat((torch.zeros_like(rgb[:, :1]), rgb), dim=1)
        for index, block in enumerate(self.blocks):
            values = block(values)
            if index == 0:
                values = self.router(values)
        return values

    @torch.no_grad()
    def export_hard(self):
        result = deepcopy(self)
        if isinstance(result.router, LearnableChannelPermutation):
            result.router = result.router.export()
        return result.eval()
