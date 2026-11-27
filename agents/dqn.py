"""Clean PyTorch Double DQN implementation for the seeded dinosaur runner."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .base import PolicyFn


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for Double DQN on DinoEnv 11-feature observations."""

    obs_dim: int = 11
    n_actions: int = 3
    hidden_dims: tuple[int, ...] = (128, 128)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 2_000
    train_freq: int = 4
    target_update_interval: int = 1_000
    tau: float = 1.0
    max_grad_norm: float = 1.0
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_steps: int = 40_000
    double_dqn: bool = True


class ReplayBuffer:
    """Fixed-capacity uniform replay buffer distinguishing termination from truncation."""

    def __init__(self, capacity: int, obs_dim: int, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self._rng = np.random.default_rng(int(seed))
        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._terminated = np.zeros(self.capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition; pass terminated (collision) only, not truncated."""
        self._obs[self._pos] = obs
        self._actions[self._pos] = int(action)
        self._rewards[self._pos] = float(reward)
        self._next_obs[self._pos] = next_obs
        self._terminated[self._pos] = 1.0 if terminated else 0.0
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a minibatch of transitions as tensors on ``device``."""
        indices = self._rng.integers(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self._obs[indices]).to(device),
            torch.from_numpy(self._actions[indices]).to(device),
            torch.from_numpy(self._rewards[indices]).to(device),
            torch.from_numpy(self._next_obs[indices]).to(device),
            torch.from_numpy(self._terminated[indices]).to(device),
        )


class QNetwork(nn.Module):
    """Feed-forward MLP mapping observations to per-action Q-values."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for width in hidden_dims:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Seeded Double DQN learner and deterministic PolicyFn callable."""

    def __init__(
        self,
        config: DQNConfig | None = None,
        *,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.config = DQNConfig() if config is None else config
        self.seed = int(seed)
        self.device = torch.device(device)

        torch.manual_seed(self.seed)
        self._rng = np.random.default_rng(self.seed)

        self.online_net = QNetwork(
            self.config.obs_dim, self.config.n_actions, self.config.hidden_dims
        ).to(self.device)
        self.target_net = QNetwork(
            self.config.obs_dim, self.config.n_actions, self.config.hidden_dims
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=self.config.learning_rate
        )
        self.buffer = ReplayBuffer(
            self.config.buffer_size, self.config.obs_dim, seed=self.seed
        )
        self._last_target_sync = 0

    def epsilon_by_step(self, step: int) -> float:
        """Linearly anneal epsilon from eps_start to eps_end over eps_decay_steps."""
        if self.config.eps_decay_steps <= 0:
            return float(self.config.eps_end)
        fraction = min(max(step, 0) / float(self.config.eps_decay_steps), 1.0)
        return float(
            self.config.eps_start + fraction * (self.config.eps_end - self.config.eps_start)
        )

    @torch.no_grad()
    def q_values(self, observation: np.ndarray) -> np.ndarray:
        """Return greedy Q-values [Q(RUN), Q(JUMP), Q(DUCK)] for diagnostics."""
        obs_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        q = self.online_net(obs_tensor).squeeze(0).cpu().numpy()
        return q.astype(np.float32, copy=False)

    def act(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool = True,
        step: int = 0,
    ) -> int:
        """Select an action greedily or via epsilon-greedy exploration."""
        if not deterministic:
            eps = self.epsilon_by_step(step)
            if float(self._rng.random()) < eps:
                return int(self._rng.integers(0, self.config.n_actions))
        q = self.q_values(observation)
        return int(np.argmax(q))

    def __call__(self, observation: np.ndarray) -> int:
        """Satisfy PolicyFn with deterministic greedy inference."""
        return self.act(observation, deterministic=True)

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        self.buffer.add(obs, action, reward, next_obs, terminated)

    def _sync_target_network(self) -> None:
        if self.config.tau >= 1.0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            return
        tau = float(self.config.tau)
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_net.parameters(), self.online_net.parameters(), strict=True
            ):
                target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)

    def maybe_update(self, step: int) -> dict[str, float] | None:
        """Run one gradient step if warmup is satisfied and step aligns with train_freq."""
        if len(self.buffer) < self.config.learning_starts or step % self.config.train_freq != 0:
            return None

        obs, actions, rewards, next_obs, terminated = self.buffer.sample(
            self.config.batch_size, self.device
        )
        q_all = self.online_net(obs)
        q_pred = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.online_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs).max(dim=1).values
            target = rewards + self.config.gamma * (1.0 - terminated) * next_q

        loss = F.smooth_l1_loss(q_pred, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        if self.config.tau < 1.0:
            self._sync_target_network()
        elif step - self._last_target_sync >= self.config.target_update_interval:
            self._sync_target_network()
            self._last_target_sync = step

        return {
            "loss": float(loss.item()),
            "mean_q": float(q_pred.mean().item()),
        }

    def save(
        self,
        path: str | Path,
        *,
        step: int = 0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save weights and plain-dict config cleanly for weights_only=True loading."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_dict = asdict(self.config)
        cfg_dict["hidden_dims"] = list(cfg_dict["hidden_dims"])
        payload = {
            "config": cfg_dict,
            "seed": self.seed,
            "step": int(step),
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "metadata": {} if extra_metadata is None else extra_metadata,
        }
        torch.save(payload, out_path)
        return out_path

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "DQNAgent":
        """Restore a saved DQNAgent from disk using weights_only=True."""
        payload = torch.load(Path(path), map_location=device, weights_only=True)
        cfg_data = dict(payload["config"])
        cfg_data["hidden_dims"] = tuple(int(v) for v in cfg_data["hidden_dims"])
        config = DQNConfig(**cfg_data)
        agent = cls(config, seed=int(payload.get("seed", 0)), device=device)
        agent.online_net.load_state_dict(payload["online_state_dict"])
        agent.target_net.load_state_dict(payload["target_state_dict"])
        agent.online_net.eval()
        agent.target_net.eval()
        return agent


def load_dqn_policy(path: str | Path, *, device: str = "cpu") -> PolicyFn:
    """Load a saved DQN checkpoint as a deterministic PolicyFn callable."""
    return DQNAgent.load(path, device=device)
