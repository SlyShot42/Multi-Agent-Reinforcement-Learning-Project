import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Optional


class LiquidCell(nn.Module):
    """
    Continuous-time recurrent 'liquid' cell:
        dh/dt = (-h + phi(W_in x + W_rec h + b)) / tau

    Integrated with a single Euler step:
        h_next = h + dt * dh/dt

    Args:
        input_dim:  Dimension of input x_t.
        hidden_dim: Dimension of hidden state h_t.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input and recurrent weights
        self.W_in = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.W_rec = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.Tensor(hidden_dim))

        # Learnable (log) time-constants to enforce positivity via softplus
        self.log_tau = nn.Parameter(torch.zeros(hidden_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming-like initialization for stability
        nn.init.kaiming_uniform_(self.W_in, a=5**0.5)
        nn.init.orthogonal_(self.W_rec)
        nn.init.zeros_(self.bias)
        # log_tau already initialized to 0 → tau ≈ softplus(0) ≈ 0.693

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        One integration step of the liquid dynamics.

        Args:
            x:  Tensor of shape (batch_size, input_dim).
            h:  Tensor of shape (batch_size, hidden_dim), previous hidden state.
            dt: Time step size for the Euler update.

        Returns:
            h_next: Tensor of shape (batch_size, hidden_dim).
        """
        # Ensure tau > 0
        tau = F.softplus(self.log_tau) + 1e-3  # (hidden_dim,)

        # Affine transform: W_in x + W_rec h + b
        # x: (B, I), W_in: (H, I) → (B, H)
        in_term = F.linear(x, self.W_in)  # (B, H)
        rec_term = F.linear(h, self.W_rec)  # (B, H)
        z = in_term + rec_term + self.bias  # (B, H)

        # Nonlinearity
        f = torch.tanh(z)  # (B, H)

        # Continuous-time dynamics
        # dh/dt = (-h + f) / tau  (broadcast tau over batch)
        dh_dt = (-h + f) / tau.unsqueeze(0)  # (B, H)

        # Euler integration step
        h_next = h + dt * dh_dt  # (B, H)

        return h_next


class LiquidDQN(nn.Module):
    """
    DQN-style Q-network with a Liquid Neural Network backbone.

    Architecture:
        state --(LiquidCell)--> h_t --(Linear)--> Q(s, ·)

    Args:
        state_dim:   Dimension of input state vector.
        action_dim:  Number of discrete actions.
        hidden_dim:  Dimension of the liquid hidden state.
        lr:          Learning rate for internal optimizer.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Liquid backbone
        self.cell = LiquidCell(state_dim, hidden_dim)

        # Linear head from hidden → Q-values
        self.head = nn.Linear(hidden_dim, action_dim)

        # Optimizer stored inside model for convenience
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(
        self,
        state: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through liquid cell + linear head.

        Args:
            state: Tensor of shape (batch_size, state_dim) or (state_dim,).
            h:     Optional hidden state of shape (batch_size, hidden_dim).
                   If None, initialized to zeros.
            dt:    Integration step size for the liquid dynamics.

        Returns:
            q_values: Tensor of shape (batch_size, action_dim).
            h_next:   Tensor of shape (batch_size, hidden_dim).
        """
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_dim)

        batch_size = state.size(0)
        device = state.device

        if h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Liquid update
        h_next = self.cell(state, h, dt=dt)  # (B, H)

        # Q-values
        q_values = self.head(h_next)  # (B, A)

        return q_values, h_next

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
        h: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Epsilon-greedy action selection using the liquid network.

        Args:
            state:   Tensor of shape (batch_size, state_dim) or (state_dim,).
            epsilon: Exploration rate in [0, 1].
            h:       Optional hidden state (batch_size, hidden_dim).
            dt:      Integration step size for the liquid dynamics.

        Returns:
            actions:  Long tensor of shape (batch_size,) with chosen actions.
            q_values: Tensor of shape (batch_size, action_dim).
            h_next:   Tensor of shape (batch_size, hidden_dim).
        """
        q_values, h_next = self.forward(state, h=h, dt=dt)
        batch_size, action_dim = q_values.shape
        device = q_values.device

        greedy_actions = torch.argmax(q_values, dim=-1)  # (B,)

        if epsilon <= 0.0:
            return greedy_actions, q_values, h_next

        # Exploration
        random_actions = torch.randint(0, action_dim, (batch_size,), device=device)
        do_random = torch.rand(batch_size, device=device) < epsilon
        actions = torch.where(do_random, random_actions, greedy_actions)

        return actions, q_values, h_next

    def train_step(
        self,
        batch_states: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_targets: torch.Tensor,
        dt: float = 1.0,
        detach_hidden: bool = True,
    ) -> float:
        """
        Single DQN training step on a minibatch using the liquid architecture.

        For simplicity, this treats each (s, a, target) as a single step
        with zero initial hidden state. For sequence-based training, you
        would typically unroll the cell over time and manage h externally.

        Args:
            batch_states:  Tensor (B, state_dim).
            batch_actions: Long tensor (B,) with actions taken.
            batch_targets: Tensor (B,) with target Q-values for those actions.
            dt:            Time step size for liquid dynamics.
            detach_hidden: Placeholder flag in case you later want to pass
                           non-zero h and control backprop through time.

        Returns:
            loss_value: Scalar loss as a Python float.
        """
        device = batch_states.device
        batch_size = batch_states.size(0)

        # Zero initial hidden state for each sample in the batch
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Forward: liquid dynamics + Q head
        q_values, h_next = self.forward(batch_states, h=h0, dt=dt)
        # Select Q(s, a) for the taken actions
        q_taken = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Standard MSE TD loss
        loss = F.mse_loss(q_taken, batch_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Optionally detach hidden if you later store h_next somewhere
        if detach_hidden:
            h_next = h_next.detach()

        return loss.item()
