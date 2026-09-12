"""One embodied actor: measured graph, learned cell dynamics, utility and motors.

This is a signed recurrent *latent* model, not measured membrane physiology.
Topology and normalized synapse counts are fixed. Cell excitability, leak, bias,
sensory encoding, utility selection and actuator decoding are learned together.
No raw observation-to-motor connection or external gait generator exists here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.nn import functional as F

ACTIVITIES = ("rest", "explore", "feed", "drink", "escape", "groom")


def csr_tensor(matrix: sparse.csr_matrix) -> torch.Tensor:
    matrix = matrix.astype(np.float32).tocsr()
    matrix.sort_indices()
    return torch.sparse_csr_tensor(
        torch.from_numpy(matrix.indptr.astype(np.int64)),
        torch.from_numpy(matrix.indices.astype(np.int64)),
        torch.from_numpy(matrix.data.copy()),
        size=matrix.shape,
        check_invariants=True,
    )


class _FixedAdjacencyMultiply(torch.autograd.Function):
    """State-only derivative; never constructs a dense neuron-by-neuron matrix."""

    @staticmethod
    def forward(ctx, adjacency, transpose, state):
        ctx.save_for_backward(transpose)
        return torch.sparse.mm(adjacency, state)

    @staticmethod
    def backward(ctx, grad_output):
        (transpose,) = ctx.saved_tensors
        return None, None, torch.sparse.mm(transpose, grad_output.contiguous())


class NeuralCore(nn.Module):
    def __init__(self, adjacency: sparse.csr_matrix):
        super().__init__()
        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Adjacency must be square: row=postsynaptic, column=presynaptic")
        if not np.isfinite(adjacency.data).all():
            raise ValueError("Nonfinite graph weights")
        self.neurons = adjacency.shape[0]
        self.connections = adjacency.nnz
        # Graph is external immutable data, bound to checkpoint by SHA-256.
        self.register_buffer("adjacency", csr_tensor(adjacency), persistent=False)
        self.register_buffer("transpose", csr_tensor(adjacency.T.tocsr()), persistent=False)
        self.excitability = nn.Parameter(torch.zeros(self.neurons))
        self.leak = nn.Parameter(torch.zeros(self.neurons))
        self.bias = nn.Parameter(torch.zeros(self.neurons))

    def forward(self, state, drive):
        signal = _FixedAdjacencyMultiply.apply(self.adjacency, self.transpose, state)
        gain = (0.25 + 3.75 * self.excitability.sigmoid())[:, None]
        leak = (0.02 + 0.96 * self.leak.sigmoid())[:, None]
        target = (gain * signal + self.bias[:, None] + drive).tanh()
        return state + leak * (target - state)


@dataclass
class BrainOutput:
    action: torch.Tensor
    state: torch.Tensor
    utility_logits: torch.Tensor
    utility_scores: torch.Tensor
    activity: torch.Tensor


class EmbodiedBrain(nn.Module):
    def __init__(
        self,
        adjacency,
        sensory_ids,
        descending_ids,
        motor_ids,
        observation_size: int,
        action_size: int,
        internal_steps: int = 4,
    ):
        super().__init__()
        if internal_steps < 2:
            raise ValueError("Need a sensory update and a utility-conditioned neural update")
        self.core = NeuralCore(adjacency)
        self.observation_size = observation_size
        self.action_size = action_size
        self.internal_steps = internal_steps
        self.register_buffer("observation_mean", torch.zeros(observation_size))
        self.register_buffer("observation_std", torch.ones(observation_size))
        for name, ids in (
            ("sensory_ids", sensory_ids),
            ("descending_ids", descending_ids),
            ("motor_ids", motor_ids),
        ):
            ids = np.asarray(ids, dtype=np.int64)
            if not len(ids) or len(np.unique(ids)) != len(ids):
                raise ValueError(f"{name} must be nonempty and unique")
            if ids.min() < 0 or ids.max() >= self.core.neurons:
                raise ValueError(f"Invalid {name}")
            self.register_buffer(name, torch.from_numpy(ids))
        if np.intersect1d(sensory_ids, motor_ids).size:
            raise ValueError("Sensory and motor routing must be disjoint")
        self.sensory_encoder = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.Tanh(),
            nn.Linear(128, len(sensory_ids)),
            nn.Tanh(),
        )
        self.utility_head = nn.Sequential(
            nn.LayerNorm(len(descending_ids)),
            nn.Linear(len(descending_ids), 128),
            nn.Tanh(),
            nn.Linear(128, len(ACTIVITIES)),
        )
        self.intention_encoder = nn.Linear(len(ACTIVITIES), len(descending_ids), bias=False)
        self.motor_decoder = nn.Sequential(
            nn.LayerNorm(len(motor_ids)),
            nn.Linear(len(motor_ids), 256),
            nn.Tanh(),
            nn.Linear(256, action_size),
            nn.Tanh(),
        )

    def initial_state(self, worlds: int):
        return self.core.bias.new_zeros((self.core.neurons, worlds))

    def reset_worlds(self, state, reset_mask):
        """Only episode resets clear memory; other worlds retain their own state."""
        return state * (~reset_mask.bool())[None, :]

    def forward(self, observation, state, activity_override=None):
        if observation.shape != (state.shape[1], self.observation_size):
            raise ValueError("Observation/world-state shape mismatch")
        encoded_observation = (
            (observation - self.observation_mean) / self.observation_std.clamp_min(0.05)
        ).clamp(-10, 10)
        drive = torch.zeros_like(state).index_copy(
            0,
            self.sensory_ids,
            self.sensory_encoder(encoded_observation).T,
        )
        state = self.core(state, drive)
        logits = self.utility_head(state[self.descending_ids].T)
        scores = logits.softmax(dim=-1)
        activity = scores.argmax(dim=-1) if activity_override is None else activity_override
        hard = F.one_hot(activity, len(ACTIVITIES)).to(scores.dtype)
        # Exact winner forward; straight-through gradient for supervised warm start.
        # RL may supply a sampled activity and use its categorical log probability.
        choice = hard + scores - scores.detach() if self.training else hard
        drive = drive.index_add(0, self.descending_ids, self.intention_encoder(choice).T)
        for _ in range(self.internal_steps - 1):
            state = self.core(state, drive)
        action = self.motor_decoder(state[self.motor_ids].T)
        return BrainOutput(action, state, logits, scores, activity)


def load_malecns(path: Path):
    """Reuse the already checksummed official graph; never substitute toy data."""
    adjacency = sparse.load_npz(path / "weights.npz").tocsr()
    with np.load(path / "brain.npz", allow_pickle=False) as metadata:
        superclass = metadata["superclass"]
        if len(superclass) != adjacency.shape[0]:
            raise ValueError("Graph and annotation sizes differ")
        sensory = np.flatnonzero(np.isin(superclass, ["vnc_sensory", "cb_sensory"]))
        descending = np.flatnonzero(superclass == "descending_neuron")
        motor = np.flatnonzero(np.isin(superclass, ["vnc_motor", "cb_motor"]))
    return adjacency, sensory, descending, motor
