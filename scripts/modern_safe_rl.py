"""Modernized Safe Reinforcement Learning implementation.

This is a modernized version of the original 0611.py script with:
- Updated imports (gymnasium instead of gym)
- Type hints and proper documentation
- Modern PyTorch practices
- Safety monitoring and evaluation
- Structured logging and configuration
"""

import argparse
import os
import time
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

from src.algorithms import SafeQLearning
from src.environments import make_safe_env
from src.evaluation import SafeRLEvaluator
from src.utils import set_seed, get_device


class ModernSafeQLearningAgent:
    """Modern Safe Q-Learning agent with improved architecture and monitoring."""
    
    def __init__(
        self,
        action_space: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        safety_threshold: float = 2.0,
        safety_penalty: float = -10.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize modern Safe Q-Learning agent.
        
        Args:
            action_space: Number of actions
            learning_rate: Learning rate for Q-network
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
            safety_threshold: Safety threshold for constraints
            safety_penalty: Penalty for safety violations
            device: PyTorch device
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.safety_threshold = safety_threshold
        self.safety_penalty = safety_penalty
        self.device = device or get_device()
        
        # Q-network (replacing Q-table)
        self.q_network = nn.Sequential(
            nn.Linear(4, 128),  # CartPole state dimension
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Safety statistics
        self.safety_violations = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_costs = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy policy with neural network.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_space)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update_q_value(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        safety_violation: bool = False
    ) -> Dict[str, float]:
        """Update Q-values using neural network.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            safety_violation: Whether safety constraint was violated
            
        Returns:
            Dictionary of training metrics
        """
        # Apply safety penalty
        if safety_violation:
            reward += self.safety_penalty
            self.safety_violations += 1
        
        self.total_steps += 1
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.BoolTensor([done]).to(self.device)
        
        # Compute current Q-value
        current_q_value = self.q_network(state_tensor).gather(1, action_tensor.unsqueeze(1))
        
        # Compute target Q-value
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            target_q_value = reward_tensor + self.discount_factor * next_q_values.max(1)[0] * (~done_tensor)
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_value.squeeze(), target_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if done and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return {
            "q_loss": loss.item(),
            "epsilon": self.epsilon,
            "safety_violations": self.safety_violations,
            "total_steps": self.total_steps,
        }


def train_safe_agent(config: DictConfig) -> Dict[str, any]:
    """Train the safe Q-learning agent.
    
    Args:
        config: Configuration object
        
    Returns:
        Training results
    """
    print("Starting Safe Reinforcement Learning Training")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Create safe environment
    env = make_safe_env(
        config.env.name,
        safety_threshold=config.env.safety_threshold,
        velocity_threshold=config.env.velocity_threshold,
        safety_penalty=config.env.safety_penalty,
    )
    
    # Create modern safe Q-learning agent
    agent = ModernSafeQLearningAgent(
        action_space=env.action_space.n,
        learning_rate=config.algorithm.learning_rate,
        discount_factor=config.algorithm.gamma,
        epsilon=config.algorithm.epsilon,
        epsilon_decay=config.algorithm.epsilon_decay,
        epsilon_min=config.algorithm.epsilon_min,
        safety_threshold=config.env.safety_threshold,
        safety_penalty=config.env.safety_penalty,
        device=get_device(),
    )
    
    # Create evaluator
    evaluator = SafeRLEvaluator(cost_limit=config.algorithm.cost_limit)
    
    print(f"Environment: {config.env.name}")
    print(f"Device: {agent.device}")
    print(f"Safety Threshold: {config.env.safety_threshold}")
    print(f"Safety Penalty: {config.env.safety_penalty}")
    print()
    
    # Training loop
    start_time = time.time()
    
    for episode in range(config.training.num_episodes):
        state, _ = env.reset(seed=config.seed + episode)
        done = False
        total_reward = 0
        total_cost = 0
        episode_safety_violations = 0
        episode_length = 0
        
        episode_rewards = []
        episode_costs = []
        episode_safety_violations_list = []
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract safety information
            safety_violation = info.get("safety_violation", False)
            cost = 1.0 if safety_violation else 0.0
            
            # Update Q-values
            training_metrics = agent.update_q_value(
                state, action, reward, next_state, done, safety_violation
            )
            
            # Update statistics
            total_reward += reward
            total_cost += cost
            episode_length += 1
            if safety_violation:
                episode_safety_violations += 1
            
            episode_rewards.append(reward)
            episode_costs.append(cost)
            episode_safety_violations_list.append(safety_violation)
            
            state = next_state
        
        # Evaluate episode
        evaluator.evaluate_episode(
            episode_rewards,
            episode_costs,
            episode_safety_violations_list,
            episode_length,
            "ModernSafeQLearning",
            config.seed + episode,
        )
        
        # Log progress
        if episode % config.training.log_interval == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:6.2f} | "
                  f"Cost: {total_cost:6.2f} | "
                  f"Length: {episode_length:3d} | "
                  f"Safety Violations: {episode_safety_violations:2d} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    training_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    
    eval_rewards = []
    eval_costs = []
    eval_lengths = []
    eval_safety_violations = []
    
    for eval_episode in range(config.evaluation.num_episodes):
        state, _ = env.reset(seed=config.seed + 10000 + eval_episode)
        done = False
        episode_reward = 0
        episode_cost = 0
        episode_length = 0
        episode_safety_violations = 0
        
        while not done:
            action = agent.select_action(state, training=False)  # No exploration
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            safety_violation = info.get("safety_violation", False)
            cost = 1.0 if safety_violation else 0.0
            
            episode_reward += reward
            episode_cost += cost
            episode_length += 1
            if safety_violation:
                episode_safety_violations += 1
            
            state = next_state
        
        eval_rewards.append(episode_reward)
        eval_costs.append(episode_cost)
        eval_lengths.append(episode_length)
        eval_safety_violations.append(episode_safety_violations)
    
    # Compute final metrics
    final_results = {
        "training_time": training_time,
        "mean_reward": np.mean(eval_rewards),
        "std_reward": np.std(eval_rewards),
        "mean_cost": np.mean(eval_costs),
        "std_cost": np.std(eval_costs),
        "mean_length": np.mean(eval_lengths),
        "std_length": np.std(eval_lengths),
        "mean_safety_violations": np.mean(eval_safety_violations),
        "std_safety_violations": np.std(eval_safety_violations),
        "safety_violation_rate": np.mean(eval_safety_violations) / np.mean(eval_lengths),
        "constraint_satisfaction_rate": np.mean([c <= config.algorithm.cost_limit for c in eval_costs]),
        "total_training_safety_violations": agent.safety_violations,
        "total_training_steps": agent.total_steps,
    }
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final Evaluation Results:")
    print(f"  Mean Reward: {final_results['mean_reward']:.2f} ± {final_results['std_reward']:.2f}")
    print(f"  Mean Cost: {final_results['mean_cost']:.2f} ± {final_results['std_cost']:.2f}")
    print(f"  Mean Length: {final_results['mean_length']:.2f} ± {final_results['std_length']:.2f}")
    print(f"  Mean Safety Violations: {final_results['mean_safety_violations']:.2f} ± {final_results['std_safety_violations']:.2f}")
    print(f"  Safety Violation Rate: {final_results['safety_violation_rate']:.3f}")
    print(f"  Constraint Satisfaction Rate: {final_results['constraint_satisfaction_rate']:.3f}")
    print(f"  Total Training Safety Violations: {final_results['total_training_safety_violations']}")
    print(f"  Total Training Steps: {final_results['total_training_steps']}")
    
    return final_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Modern Safe Reinforcement Learning")
    parser.add_argument("--config", type=str, default="configs/safe_qlearning.yaml", help="Config file")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override with command line arguments
    config.output_dir = args.output_dir
    config.seed = args.seed
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Train agent
    results = train_safe_agent(config)
    
    # Save results
    results_path = os.path.join(config.output_dir, "modern_safe_rl_results.pt")
    torch.save(results, results_path)
    print(f"\nResults saved to: {results_path}")
    
    # Safety disclaimer
    print("\n" + "=" * 50)
    print("SAFETY DISCLAIMER")
    print("=" * 50)
    print("This implementation is for research and educational purposes only.")
    print("It should NOT be used for controlling real-world systems,")
    print("especially in safety-critical domains such as:")
    print("- Autonomous vehicles")
    print("- Medical devices")
    print("- Industrial control systems")
    print("- Financial trading systems")
    print("- Any system where failure could result in harm")


if __name__ == "__main__":
    main()
