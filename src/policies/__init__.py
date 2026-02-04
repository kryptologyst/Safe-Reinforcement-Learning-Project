"""Policy networks for safe reinforcement learning."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from ..utils import create_mlp


class SafeQNetwork(nn.Module):
    """Safe Q-Network with safety-aware architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (256, 256)):
        """Initialize Safe Q-Network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Main Q-network
        self.q_network = create_mlp(
            state_dim, action_dim, hidden_dims, nn.ReLU, None
        )
        
        # Safety head (optional)
        self.safety_head = create_mlp(
            state_dim, 1, hidden_dims, nn.ReLU, nn.Sigmoid
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            state: State tensor
            
        Returns:
            Q-values
        """
        return self.q_network(state)
    
    def get_safety_score(self, state: torch.Tensor) -> torch.Tensor:
        """Get safety score for state.
        
        Args:
            state: State tensor
            
        Returns:
            Safety score (0-1)
        """
        return self.safety_head(state)


class SafeActorCritic(nn.Module):
    """Safe Actor-Critic network with safety-aware architecture."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: Tuple[int, ...] = (256, 256),
        continuous: bool = False,
    ):
        """Initialize Safe Actor-Critic.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            continuous: Whether action space is continuous
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        
        # Shared feature extractor
        self.feature_extractor = create_mlp(
            state_dim, hidden_dims[-1], hidden_dims[:-1], nn.ReLU, None
        )
        
        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
            self.actor_log_std = nn.Linear(hidden_dims[-1], action_dim)
        else:
            self.actor = nn.Linear(hidden_dims[-1], action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dims[-1], 1)
        
        # Safety head
        self.safety_head = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        features = self.feature_extractor(state)
        
        if self.continuous:
            # Continuous action space
            mean = self.actor_mean(features)
            log_std = self.actor_log_std(features)
            log_std = torch.clamp(log_std, min=-20, max=2)
            std = torch.exp(log_std)
            
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Clamp action to valid range
            action = torch.tanh(action)
        else:
            # Discrete action space
            logits = self.actor(features)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        value = self.critic(features).squeeze(-1)
        
        return action, log_prob, value
    
    def get_safety_score(self, state: torch.Tensor) -> torch.Tensor:
        """Get safety score for state.
        
        Args:
            state: State tensor
            
        Returns:
            Safety score (0-1)
        """
        features = self.feature_extractor(state)
        return torch.sigmoid(self.safety_head(features)).squeeze(-1)
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for given states.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        features = self.feature_extractor(state)
        
        if self.continuous:
            mean = self.actor_mean(features)
            log_std = self.actor_log_std(features)
            log_std = torch.clamp(log_std, min=-20, max=2)
            std = torch.exp(log_std)
            
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            logits = self.actor(features)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        
        value = self.critic(features).squeeze(-1)
        
        return log_prob, value, entropy


class SafePolicyNetwork(nn.Module):
    """Safe policy network with safety constraints."""
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dims: Tuple[int, ...] = (256, 256),
        continuous: bool = False,
        safety_layers: int = 2,
    ):
        """Initialize Safe Policy Network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            continuous: Whether action space is continuous
            safety_layers: Number of safety constraint layers
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.safety_layers = safety_layers
        
        # Main policy network
        self.policy_net = create_mlp(
            state_dim, hidden_dims[-1], hidden_dims[:-1], nn.ReLU, None
        )
        
        # Action head
        if continuous:
            self.action_mean = nn.Linear(hidden_dims[-1], action_dim)
            self.action_log_std = nn.Linear(hidden_dims[-1], action_dim)
        else:
            self.action_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Safety constraint layers
        self.safety_constraints = nn.ModuleList([
            nn.Linear(hidden_dims[-1], hidden_dims[-1]) 
            for _ in range(safety_layers)
        ])
        
        # Safety output
        self.safety_output = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action, safety_score)
        """
        features = self.policy_net(state)
        
        # Apply safety constraints
        safety_features = features
        for constraint_layer in self.safety_constraints:
            safety_features = F.relu(constraint_layer(safety_features))
        
        # Get safety score
        safety_score = torch.sigmoid(self.safety_output(safety_features))
        
        # Get action
        if self.continuous:
            mean = self.action_mean(features)
            log_std = self.action_log_std(features)
            log_std = torch.clamp(log_std, min=-20, max=2)
            std = torch.exp(log_std)
            
            dist = Normal(mean, std)
            action = dist.sample()
            
            # Apply safety constraints to action
            action = action * safety_score.unsqueeze(-1)
            action = torch.tanh(action)
        else:
            logits = self.action_head(features)
            
            # Apply safety constraints to logits
            logits = logits * safety_score.unsqueeze(-1)
            
            dist = Categorical(logits=logits)
            action = dist.sample()
        
        return action, safety_score
    
    def get_log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get log probability of action given state.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Log probability
        """
        features = self.policy_net(state)
        
        if self.continuous:
            mean = self.action_mean(features)
            log_std = self.action_log_std(features)
            log_std = torch.clamp(log_std, min=-20, max=2)
            std = torch.exp(log_std)
            
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            logits = self.action_head(features)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(action)
        
        return log_prob
