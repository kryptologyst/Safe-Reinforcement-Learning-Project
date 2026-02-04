"""Core utilities for safe reinforcement learning."""

import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Tuple[int, ...] = (256, 256),
    activation: nn.Module = nn.ReLU,
    output_activation: Optional[nn.Module] = None,
) -> nn.Module:
    """Create a multi-layer perceptron.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: Hidden layer dimensions
        activation: Activation function
        output_activation: Output activation function
        
    Returns:
        MLP network
    """
    layers = []
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:  # Don't add activation after last layer
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    
    return nn.Sequential(*layers)


def safe_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely convert tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Numpy array
    """
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Reward tensor
        values: Value estimates
        next_values: Next state value estimates
        dones: Done flags
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = torch.zeros_like(rewards)
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t] if not dones[t] else 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns


def compute_cvar(values: np.ndarray, alpha: float = 0.05) -> float:
    """Compute Conditional Value at Risk (CVaR).
    
    Args:
        values: Array of values
        alpha: Risk level (e.g., 0.05 for 5% tail)
        
    Returns:
        CVaR value
    """
    sorted_values = np.sort(values)
    tail_size = int(len(values) * alpha)
    if tail_size == 0:
        return np.min(values)
    return np.mean(sorted_values[:tail_size])


class RunningStats:
    """Running statistics for normalization."""
    
    def __init__(self, shape: Tuple[int, ...]) -> None:
        """Initialize running statistics.
        
        Args:
            shape: Shape of the statistics
        """
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.std = np.ones(shape, dtype=np.float64)
    
    def update(self, x: np.ndarray) -> None:
        """Update statistics with new data.
        
        Args:
            x: New data point
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.n + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.n
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.n * batch_count / total_count
        new_var = m2 / total_count
        
        self.n = total_count
        self.mean = new_mean
        self.var = new_var
        self.std = np.sqrt(self.var + 1e-8)
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize data using running statistics.
        
        Args:
            x: Data to normalize
            
        Returns:
            Normalized data
        """
        return (x - self.mean) / self.std


def load_config(config_path: str) -> DictConfig:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    from omegaconf import OmegaConf
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration object
        config_path: Path to save configuration
    """
    from omegaconf import OmegaConf
    OmegaConf.save(config, config_path)
