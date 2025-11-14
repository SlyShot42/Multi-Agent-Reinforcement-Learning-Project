import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple


class DQN(nn.Module):
    """
    Simple MLP-based Q-network for discrete-action reinforcement learning.

    Args:
        state_dim: Dimension of the (flattened) state/input vector.
        action_dim: Number of discrete actions.
        hidden_sizes: Sizes of hidden layers (e.g., [128, 128]).
        lr: Learning rate for the internal optimizer.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int] = (128, 128),
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        # Last layer outputs one Q-value per action
        layers.append(nn.Linear(input_dim, action_dim))

        self.net = nn.Sequential(*layers)

        # Optimizer lives inside the model for convenience
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: Tensor of shape (batch_size, state_dim).

        Returns:
            Q-values tensor of shape (batch_size, action_dim).
        """
        return self.net(state)

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Epsilon-greedy action selection.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).
            epsilon: Exploration rate in [0, 1].

        Returns:
            actions: Long tensor of shape (batch_size,) with chosen actions.
            q_values: Tensor of shape (batch_size, action_dim) with Q-values.
        """
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)

        q_values = self.forward(state)  # (B, A)
        batch_size, action_dim = q_values.shape

        # Greedy actions
        greedy_actions = torch.argmax(q_values, dim=-1)  # (B,)

        if epsilon <= 0.0:
            return greedy_actions, q_values

        # Exploration: with prob epsilon choose random action
        random_actions = torch.randint(
            0, action_dim, (batch_size,), device=state.device
        )

        # Bernoulli mask: 1 -> random, 0 -> greedy
        do_random = torch.rand(batch_size, device=state.device) < epsilon
        actions = torch.where(do_random, random_actions, greedy_actions)

        return actions, q_values

    def train_step(
        self,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_targets: torch.Tensor,
    ) -> float:
        """
        Single DQN training step on a minibatch.

        Args:
            batch_states: Tensor of shape (batch_size, state_dim).
            batch_actions: Long tensor of shape (batch_size,) with actions taken.
            batch_targets: Tensor of shape (batch_size,) with target Q-values
                           (e.g., r + γ max_a' Q_target(s', a')).

        Returns:
            loss_value: Scalar loss as a Python float.
        """
        # Forward pass
        q_values = self.forward(batch_states)  # (B, A)
        q_taken = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Compute loss
        loss = F.mse_loss(q_taken, batch_targets)

        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
