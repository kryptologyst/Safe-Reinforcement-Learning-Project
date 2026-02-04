"""Tests for safe reinforcement learning."""

import pytest
import numpy as np
import torch
import gymnasium as gym

from src.algorithms import SafeQLearning, ConstrainedPolicyOptimization, LagrangianSafeRL
from src.environments import make_safe_env, SafeCartPoleWrapper
from src.policies import SafeQNetwork, SafeActorCritic
from src.evaluation import SafeRLEvaluator
from src.utils import set_seed, get_device, compute_gae, compute_cvar


class TestSafeQLearning:
    """Test Safe Q-Learning algorithm."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        agent = SafeQLearning(state_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.epsilon == 1.0
    
    def test_action_selection(self):
        """Test action selection."""
        agent = SafeQLearning(state_dim=4, action_dim=2)
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        # Test exploration
        agent.epsilon = 1.0
        action = agent.select_action(state, training=True)
        assert action in [0, 1]
        
        # Test exploitation
        agent.epsilon = 0.0
        action = agent.select_action(state, training=True)
        assert action in [0, 1]
    
    def test_update(self):
        """Test Q-value update."""
        agent = SafeQLearning(state_dim=4, action_dim=2)
        
        batch = {
            "states": torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]]),
            "actions": torch.LongTensor([0]),
            "rewards": torch.FloatTensor([1.0]),
            "next_states": torch.FloatTensor([[0.2, 0.3, 0.4, 0.5]]),
            "dones": torch.BoolTensor([False]),
            "safety_violations": torch.FloatTensor([0.0]),
        }
        
        metrics = agent.update(batch)
        assert "q_loss" in metrics
        assert "epsilon" in metrics


class TestConstrainedPolicyOptimization:
    """Test Constrained Policy Optimization algorithm."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        agent = ConstrainedPolicyOptimization(state_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.cost_limit == 0.01
    
    def test_action_selection(self):
        """Test action selection."""
        agent = ConstrainedPolicyOptimization(state_dim=4, action_dim=2)
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        action, log_prob, value = agent.select_action(state, training=True)
        assert action in [0, 1]
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_update(self):
        """Test policy update."""
        agent = ConstrainedPolicyOptimization(state_dim=4, action_dim=2)
        
        batch = {
            "states": torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]]),
            "actions": torch.LongTensor([0]),
            "rewards": torch.FloatTensor([1.0]),
            "costs": torch.FloatTensor([0.0]),
            "old_log_probs": torch.FloatTensor([0.0]),
            "old_values": torch.FloatTensor([0.0]),
            "advantages": torch.FloatTensor([0.0]),
            "returns": torch.FloatTensor([1.0]),
            "dones": torch.BoolTensor([False]),
        }
        
        metrics = agent.update(batch)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "cost_loss" in metrics


class TestLagrangianSafeRL:
    """Test Lagrangian Safe RL algorithm."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        agent = LagrangianSafeRL(state_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert agent.cost_limit == 0.01
        assert agent.lagrangian_multiplier.item() == 1.0
    
    def test_action_selection(self):
        """Test action selection."""
        agent = LagrangianSafeRL(state_dim=4, action_dim=2)
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        action, log_prob, value = agent.select_action(state, training=True)
        assert action in [0, 1]
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_update(self):
        """Test policy update."""
        agent = LagrangianSafeRL(state_dim=4, action_dim=2)
        
        batch = {
            "states": torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]]),
            "actions": torch.LongTensor([0]),
            "rewards": torch.FloatTensor([1.0]),
            "costs": torch.FloatTensor([0.0]),
            "old_log_probs": torch.FloatTensor([0.0]),
            "old_values": torch.FloatTensor([0.0]),
            "advantages": torch.FloatTensor([0.0]),
            "returns": torch.FloatTensor([1.0]),
            "dones": torch.BoolTensor([False]),
        }
        
        metrics = agent.update(batch)
        assert "policy_loss" in metrics
        assert "lagrangian_loss" in metrics
        assert "lagrangian_multiplier" in metrics


class TestSafeEnvironments:
    """Test safe environment wrappers."""
    
    def test_safe_cartpole_wrapper(self):
        """Test SafeCartPoleWrapper."""
        env = gym.make("CartPole-v1")
        safe_env = SafeCartPoleWrapper(env)
        
        # Test reset
        obs, info = safe_env.reset()
        assert len(obs) == 5  # Original 4 + safety flag
        assert "safety_violation" in info
        
        # Test step
        action = 0
        obs, reward, terminated, truncated, info = safe_env.step(action)
        assert len(obs) == 5
        assert "safety_violation" in info
        assert "cart_position" in info
        assert "cart_velocity" in info
    
    def test_make_safe_env(self):
        """Test make_safe_env function."""
        env = make_safe_env("CartPole-v1")
        assert env.observation_space.shape[0] == 5  # Original 4 + safety flag
        
        obs, info = env.reset()
        assert len(obs) == 5
        assert "safety_violation" in info


class TestPolicies:
    """Test policy networks."""
    
    def test_safe_q_network(self):
        """Test SafeQNetwork."""
        network = SafeQNetwork(state_dim=4, action_dim=2)
        state = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])
        
        q_values = network(state)
        assert q_values.shape == (1, 2)
        
        safety_score = network.get_safety_score(state)
        assert safety_score.shape == (1, 1)
        assert 0 <= safety_score.item() <= 1
    
    def test_safe_actor_critic(self):
        """Test SafeActorCritic."""
        network = SafeActorCritic(state_dim=4, action_dim=2)
        state = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])
        
        action, log_prob, value = network(state)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)
        
        safety_score = network.get_safety_score(state)
        assert safety_score.shape == (1,)
        assert 0 <= safety_score.item() <= 1


class TestEvaluation:
    """Test evaluation utilities."""
    
    def test_safe_rl_evaluator(self):
        """Test SafeRLEvaluator."""
        evaluator = SafeRLEvaluator(cost_limit=0.01, alpha=0.05)
        
        # Test episode evaluation
        episode_rewards = [1.0, 2.0, 3.0]
        episode_costs = [0.0, 0.005, 0.01]
        episode_safety_violations = [False, False, True]
        episode_length = 3
        
        metrics = evaluator.evaluate_episode(
            episode_rewards, episode_costs, episode_safety_violations,
            episode_length, "test_algorithm", 42
        )
        
        assert "total_reward" in metrics
        assert "total_cost" in metrics
        assert "safety_violations" in metrics
        assert "constraint_satisfaction" in metrics
    
    def test_algorithm_evaluation(self):
        """Test algorithm evaluation."""
        evaluator = SafeRLEvaluator()
        
        # Add some test results
        for i in range(10):
            evaluator.evaluate_episode(
                [1.0] * 5, [0.0] * 5, [False] * 5,
                5, "test_algorithm", 42
            )
        
        metrics = evaluator.evaluate_algorithm("test_algorithm")
        assert "mean_reward" in metrics
        assert "mean_cost" in metrics
        assert "constraint_satisfaction_rate" in metrics


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_compute_gae(self):
        """Test GAE computation."""
        rewards = torch.FloatTensor([1.0, 2.0, 3.0])
        values = torch.FloatTensor([0.5, 1.0, 1.5])
        next_values = torch.FloatTensor([1.0, 1.5, 2.0])
        dones = torch.BoolTensor([False, False, True])
        
        advantages, returns = compute_gae(rewards, values, next_values, dones)
        
        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
    
    def test_compute_cvar(self):
        """Test CVaR computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cvar = compute_cvar(values, alpha=0.2)
        
        # CVaR should be less than or equal to the mean
        assert cvar <= np.mean(values)
        # CVaR should be greater than or equal to the minimum
        assert cvar >= np.min(values)


if __name__ == "__main__":
    pytest.main([__file__])
