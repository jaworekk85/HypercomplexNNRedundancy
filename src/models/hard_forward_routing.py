"""Hard permutation forward, Sinkhorn surrogate backward (not a novel STE).

References: Bengio2013STE, Mena2018, Wang2023PSTE in references.bib. This
deterministic application is not a reproduction of their complete methods.
"""

import numpy as np
import torch
from torch.nn import functional as F

from src.models.channel_routing import LearnableChannelPermutation, RoutedCifarCNN, maximum_assignment


class HardForwardChannelPermutation(LearnableChannelPermutation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("assignment_switch_events", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("assignment_switched_positions", torch.tensor(0, dtype=torch.int64))

    @torch.no_grad()
    def refresh_assignment(self):
        chosen = maximum_assignment(self.logits.detach().cpu().double().numpy())
        changed = int(np.sum(chosen != self.indices.detach().cpu().numpy()))
        if changed:
            self.assignment_switch_events.add_(1)
            self.assignment_switched_positions.add_(changed)
            self.indices.copy_(torch.as_tensor(chosen, device=self.indices.device))

    def forward(self, inputs):
        if inputs.ndim != 4 or inputs.shape[1] != len(self.indices):
            raise ValueError("Unexpected hard routing shape")
        if bool(self.hard_enabled):
            return inputs.index_select(1, self.indices)
        self.refresh_assignment()
        if not torch.is_grad_enabled():
            return inputs.index_select(1, self.indices)
        soft = self.soft_matrix()
        hard = F.one_hot(self.indices, num_classes=len(self.indices)).to(soft.dtype)
        # Parentheses are essential: forward value is exactly 0/1. Input
        # gradients use H, while logits receive the biased Sinkhorn surrogate.
        # Finite-difference gradcheck of the discrete map would be inappropriate.
        matrix = hard + (soft - soft.detach())
        return torch.einsum("ij,njhw->nihw", matrix.to(inputs.dtype), inputs)

    @torch.no_grad()
    def harden(self):
        self.refresh_assignment()
        super().harden()


class HardForwardRoutedCNN(RoutedCifarCNN):
    def __init__(self, config):
        super().__init__("learned", config["seed"], config["quaternion_widths"], config["dropout"],
                         config["initial_strength"], config["temperature_start"], config["sinkhorn_iterations"])
        # Both constructors use private generators; replacing the router does
        # not consume the global backbone/dropout training RNG stream.
        self.router = HardForwardChannelPermutation(4*config["quaternion_widths"][0], config["seed"]+60000,
                                                    config["initial_strength"], config["temperature_start"], config["sinkhorn_iterations"])


def make_model(config, mode):
    if mode == "learned":
        return HardForwardRoutedCNN(config)
    return RoutedCifarCNN(mode, config["seed"], config["quaternion_widths"], config["dropout"],
                          config["initial_strength"], config["temperature_start"], config["sinkhorn_iterations"])
