"""Safe reinforcement learning algorithms."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from ..utils import compute_gae, get_device, safe_tensor_to_numpy
from .policies import SafeActorCritic, SafeQNetwork


class SafeQLearning:
    """Safe Q-Learning with penalty-based reward shaping."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        safety_threshold: float = 2.0,
        safety_penalty: float = -10.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize Safe Q-Learning agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            safety_threshold: Safety threshold for constraints
            safety_penalty: Penalty for safety violations
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.safety_threshold = safety_threshold
        self.safety_penalty = safety_penalty
        self.device = device or get_device()
        
        # Q-network
        self.q_network = SafeQNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Safety statistics
        self.safety_violations = 0
        self.total_steps = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update Q-network with batch of experiences.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Apply safety penalties
        safety_violations = batch.get("safety_violations", torch.zeros_like(rewards))
        rewards = rewards + safety_violations * self.safety_penalty
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            target_q_values = rewards + self.gamma * next_q_values.max(1)[0] * (1 - dones)
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            "q_loss": loss.item(),
            "epsilon": self.epsilon,
            "safety_violations": safety_violations.sum().item(),
        }


class ConstrainedPolicyOptimization:
    """Constrained Policy Optimization (CPO) algorithm."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        cost_limit: float = 0.01,
        cpo_iters: int = 10,
        cpo_step_size: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        """Initialize CPO agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            lam: GAE lambda
            clip_ratio: PPO clip ratio
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            cost_limit: Cost limit for constraints
            cpo_iters: CPO optimization iterations
            cpo_step_size: CPO step size
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.cost_limit = cost_limit
        self.cpo_iters = cpo_iters
        self.cpo_step_size = cpo_step_size
        self.device = device or get_device()
        
        # Networks
        self.policy = SafeActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Safety statistics
        self.safety_violations = 0
        self.total_steps = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action using current policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy(state_tensor)
            return action.item(), log_prob.item(), value.item()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using CPO.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        costs = batch.get("costs", torch.zeros_like(rewards)).to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        old_values = batch["old_values"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            _, _, values = self.policy(states)
            adv, ret = compute_gae(rewards, old_values, values, batch["dones"], self.gamma, self.lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # CPO optimization
        for _ in range(self.cpo_iters):
            # Forward pass
            _, log_probs, values = self.policy(states)
            
            # Compute policy loss (PPO)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = F.mse_loss(values, returns)
            
            # Compute entropy loss
            entropy_loss = -log_probs.mean()
            
            # Compute cost loss (constraint)
            cost_loss = costs.mean()
            
            # Total loss
            total_loss = (
                policy_loss + 
                self.value_loss_coef * value_loss + 
                self.entropy_coef * entropy_loss
            )
            
            # Check constraint satisfaction
            if cost_loss <= self.cost_limit:
                # Unconstrained update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
            else:
                # Constrained update (simplified CPO)
                # In practice, this would involve solving a constrained optimization problem
                # Here we use a penalty method as a simplification
                penalty_loss = total_loss + 1000 * torch.relu(cost_loss - self.cost_limit)
                self.optimizer.zero_grad()
                penalty_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "cost_loss": cost_loss.item(),
            "total_loss": total_loss.item(),
            "safety_violations": costs.sum().item(),
        }


class LagrangianSafeRL:
    """Lagrangian method for safe reinforcement learning."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        lagrangian_lr: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        cost_limit: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        """Initialize Lagrangian Safe RL agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            lagrangian_lr: Lagrangian multiplier learning rate
            gamma: Discount factor
            lam: GAE lambda
            clip_ratio: PPO clip ratio
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            cost_limit: Cost limit for constraints
            device: PyTorch device
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.cost_limit = cost_limit
        self.device = device or get_device()
        
        # Networks
        self.policy = SafeActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Lagrangian multiplier
        self.lagrangian_multiplier = torch.tensor(1.0, requires_grad=True, device=self.device)
        self.lagrangian_optimizer = torch.optim.Adam([self.lagrangian_multiplier], lr=lagrangian_lr)
        
        # Safety statistics
        self.safety_violations = 0
        self.total_steps = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """Select action using current policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy(state_tensor)
            return action.item(), log_prob.item(), value.item()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy using Lagrangian method.
        
        Args:
            batch: Batch of experiences
            
        Returns:
            Dictionary of training metrics
        """
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        costs = batch.get("costs", torch.zeros_like(rewards)).to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        old_values = batch["old_values"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            _, _, values = self.policy(states)
            adv, ret = compute_gae(rewards, old_values, values, batch["dones"], self.gamma, self.lam)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        _, log_probs, values = self.policy(states)
        
        # Compute policy loss (PPO)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = F.mse_loss(values, returns)
        
        # Compute entropy loss
        entropy_loss = -log_probs.mean()
        
        # Compute cost loss (constraint)
        cost_loss = costs.mean()
        
        # Lagrangian loss
        lagrangian_loss = self.lagrangian_multiplier * (cost_loss - self.cost_limit)
        
        # Total loss
        total_loss = (
            policy_loss + 
            self.value_loss_coef * value_loss + 
            self.entropy_coef * entropy_loss +
            lagrangian_loss
        )
        
        # Update policy
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Update Lagrangian multiplier
        self.lagrangian_optimizer.zero_grad()
        (-lagrangian_loss).backward()  # Maximize Lagrangian multiplier
        self.lagrangian_optimizer.step()
        
        # Ensure Lagrangian multiplier is non-negative
        with torch.no_grad():
            self.lagrangian_multiplier.clamp_(min=0.0)
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "cost_loss": cost_loss.item(),
            "lagrangian_loss": lagrangian_loss.item(),
            "lagrangian_multiplier": self.lagrangian_multiplier.item(),
            "total_loss": total_loss.item(),
            "safety_violations": costs.sum().item(),
        }
