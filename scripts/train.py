"""Training scripts for safe reinforcement learning."""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ..algorithms import ConstrainedPolicyOptimization, LagrangianSafeRL, SafeQLearning
from ..environments import make_safe_env
from ..evaluation import SafeRLEvaluator
from ..utils import set_seed, get_device, compute_gae


class SafeRLTrainer:
    """Trainer for safe reinforcement learning algorithms."""
    
    def __init__(self, config: DictConfig):
        """Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.device = get_device()
        
        # Set random seeds
        set_seed(config.seed)
        
        # Create environment
        self.env = make_safe_env(
            config.env.name,
            safety_threshold=config.env.safety_threshold,
            velocity_threshold=config.env.velocity_threshold,
            safety_penalty=config.env.safety_penalty,
        )
        
        # Create algorithm
        self.algorithm = self._create_algorithm()
        
        # Create evaluator
        self.evaluator = SafeRLEvaluator(
            cost_limit=config.algorithm.cost_limit,
            alpha=config.evaluation.alpha,
        )
        
        # Training statistics
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_lengths = []
        self.safety_violations = []
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _create_algorithm(self):
        """Create algorithm based on config.
        
        Returns:
            Algorithm instance
        """
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        if self.config.algorithm.name == "safe_qlearning":
            return SafeQLearning(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=self.config.algorithm.learning_rate,
                gamma=self.config.algorithm.gamma,
                epsilon=self.config.algorithm.epsilon,
                epsilon_decay=self.config.algorithm.epsilon_decay,
                epsilon_min=self.config.algorithm.epsilon_min,
                safety_threshold=self.config.env.safety_threshold,
                safety_penalty=self.config.env.safety_penalty,
                device=self.device,
            )
        elif self.config.algorithm.name == "cpo":
            return ConstrainedPolicyOptimization(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=tuple(self.config.algorithm.hidden_dims),
                learning_rate=self.config.algorithm.learning_rate,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                clip_ratio=self.config.algorithm.clip_ratio,
                value_loss_coef=self.config.algorithm.value_loss_coef,
                entropy_coef=self.config.algorithm.entropy_coef,
                max_grad_norm=self.config.algorithm.max_grad_norm,
                cost_limit=self.config.algorithm.cost_limit,
                cpo_iters=self.config.algorithm.cpo_iters,
                cpo_step_size=self.config.algorithm.cpo_step_size,
                device=self.device,
            )
        elif self.config.algorithm.name == "lagrangian":
            return LagrangianSafeRL(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=tuple(self.config.algorithm.hidden_dims),
                learning_rate=self.config.algorithm.learning_rate,
                lagrangian_lr=self.config.algorithm.lagrangian_lr,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                clip_ratio=self.config.algorithm.clip_ratio,
                value_loss_coef=self.config.algorithm.value_loss_coef,
                entropy_coef=self.config.algorithm.entropy_coef,
                max_grad_norm=self.config.algorithm.max_grad_norm,
                cost_limit=self.config.algorithm.cost_limit,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm.name}")
    
    def train(self) -> Dict[str, Any]:
        """Train the algorithm.
        
        Returns:
            Dictionary of training results
        """
        print(f"Training {self.config.algorithm.name} on {self.config.env.name}")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.config.output_dir}")
        
        start_time = time.time()
        
        # Training loop
        for episode in tqdm(range(self.config.training.num_episodes), desc="Training"):
            episode_result = self._train_episode(episode)
            
            # Log episode results
            if episode % self.config.training.log_interval == 0:
                self._log_episode(episode, episode_result)
            
            # Evaluate periodically
            if episode % self.config.training.eval_interval == 0:
                self._evaluate(episode)
            
            # Save checkpoint
            if episode % self.config.training.save_interval == 0:
                self._save_checkpoint(episode)
        
        training_time = time.time() - start_time
        
        # Final evaluation
        final_results = self._evaluate(self.config.training.num_episodes)
        
        # Save final results
        self._save_results(final_results, training_time)
        
        return final_results
    
    def _train_episode(self, episode: int) -> Dict[str, Any]:
        """Train for one episode.
        
        Args:
            episode: Episode number
            
        Returns:
            Episode results
        """
        state, _ = self.env.reset(seed=self.config.seed + episode)
        done = False
        episode_reward = 0
        episode_cost = 0
        episode_length = 0
        episode_safety_violations = 0
        
        # Store experience for batch training
        states, actions, rewards, costs, dones, old_log_probs, old_values = [], [], [], [], [], [], []
        
        while not done:
            # Select action
            if self.config.algorithm.name == "safe_qlearning":
                action = self.algorithm.select_action(state, training=True)
                old_log_prob = 0.0  # Q-learning doesn't use log probs
                old_value = 0.0  # Q-learning doesn't use values
            else:
                action, old_log_prob, old_value = self.algorithm.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Extract safety information
            cost = info.get("safety_violation", False) * 1.0
            safety_violation = info.get("safety_violation", False)
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            costs.append(cost)
            dones.append(done)
            old_log_probs.append(old_log_prob)
            old_values.append(old_value)
            
            # Update statistics
            episode_reward += reward
            episode_cost += cost
            episode_length += 1
            if safety_violation:
                episode_safety_violations += 1
            
            state = next_state
        
        # Batch training
        if len(states) > 0:
            batch = self._create_batch(states, actions, rewards, costs, dones, old_log_probs, old_values)
            training_metrics = self.algorithm.update(batch)
        else:
            training_metrics = {}
        
        # Store episode results
        episode_result = {
            "episode": episode,
            "reward": episode_reward,
            "cost": episode_cost,
            "length": episode_length,
            "safety_violations": episode_safety_violations,
            "training_metrics": training_metrics,
        }
        
        self.episode_rewards.append(episode_reward)
        self.episode_costs.append(episode_cost)
        self.episode_lengths.append(episode_length)
        self.safety_violations.append(episode_safety_violations)
        
        return episode_result
    
    def _create_batch(self, states, actions, rewards, costs, dones, old_log_probs, old_values):
        """Create batch for training.
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            costs: List of costs
            dones: List of done flags
            old_log_probs: List of old log probabilities
            old_values: List of old values
            
        Returns:
            Batch dictionary
        """
        batch = {
            "states": torch.FloatTensor(np.array(states)),
            "actions": torch.LongTensor(actions),
            "rewards": torch.FloatTensor(rewards),
            "costs": torch.FloatTensor(costs),
            "dones": torch.BoolTensor(dones),
            "old_log_probs": torch.FloatTensor(old_log_probs),
            "old_values": torch.FloatTensor(old_values),
        }
        
        # Compute advantages and returns for policy gradient methods
        if self.config.algorithm.name in ["cpo", "lagrangian"]:
            advantages, returns = compute_gae(
                batch["rewards"],
                batch["old_values"],
                batch["old_values"],  # Simplified - would need next values in practice
                batch["dones"],
                self.config.algorithm.gamma,
                self.config.algorithm.lam,
            )
            batch["advantages"] = advantages
            batch["returns"] = returns
        
        return batch
    
    def _log_episode(self, episode: int, episode_result: Dict[str, Any]) -> None:
        """Log episode results.
        
        Args:
            episode: Episode number
            episode_result: Episode results
        """
        print(f"Episode {episode}:")
        print(f"  Reward: {episode_result['reward']:.2f}")
        print(f"  Cost: {episode_result['cost']:.2f}")
        print(f"  Length: {episode_result['length']}")
        print(f"  Safety Violations: {episode_result['safety_violations']}")
        
        if episode_result['training_metrics']:
            for key, value in episode_result['training_metrics'].items():
                print(f"  {key}: {value:.4f}")
        print()
    
    def _evaluate(self, episode: int) -> Dict[str, Any]:
        """Evaluate the current policy.
        
        Args:
            episode: Episode number
            
        Returns:
            Evaluation results
        """
        print(f"Evaluating at episode {episode}...")
        
        eval_rewards = []
        eval_costs = []
        eval_lengths = []
        eval_safety_violations = []
        
        for eval_episode in range(self.config.evaluation.num_episodes):
            state, _ = self.env.reset(seed=self.config.seed + 10000 + eval_episode)
            done = False
            episode_reward = 0
            episode_cost = 0
            episode_length = 0
            episode_safety_violations = 0
            
            while not done:
                if self.config.algorithm.name == "safe_qlearning":
                    action = self.algorithm.select_action(state, training=False)
                else:
                    action, _, _ = self.algorithm.select_action(state, training=False)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                cost = info.get("safety_violation", False) * 1.0
                safety_violation = info.get("safety_violation", False)
                
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
        
        # Compute evaluation metrics
        eval_results = {
            "episode": episode,
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "mean_cost": np.mean(eval_costs),
            "std_cost": np.std(eval_costs),
            "mean_length": np.mean(eval_lengths),
            "std_length": np.std(eval_lengths),
            "mean_safety_violations": np.mean(eval_safety_violations),
            "std_safety_violations": np.std(eval_safety_violations),
        }
        
        print(f"Evaluation Results:")
        print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Mean Cost: {eval_results['mean_cost']:.2f} ± {eval_results['std_cost']:.2f}")
        print(f"  Mean Length: {eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}")
        print(f"  Mean Safety Violations: {eval_results['mean_safety_violations']:.2f} ± {eval_results['std_safety_violations']:.2f}")
        print()
        
        return eval_results
    
    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint.
        
        Args:
            episode: Episode number
        """
        checkpoint_path = os.path.join(self.config.output_dir, f"checkpoint_episode_{episode}.pt")
        
        checkpoint = {
            "episode": episode,
            "algorithm_state": self.algorithm.__dict__,
            "episode_rewards": self.episode_rewards,
            "episode_costs": self.episode_costs,
            "episode_lengths": self.episode_lengths,
            "safety_violations": self.safety_violations,
            "config": self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_results(self, final_results: Dict[str, Any], training_time: float) -> None:
        """Save final results.
        
        Args:
            final_results: Final evaluation results
            training_time: Total training time
        """
        results = {
            "config": self.config,
            "final_results": final_results,
            "training_time": training_time,
            "episode_rewards": self.episode_rewards,
            "episode_costs": self.episode_costs,
            "episode_lengths": self.episode_lengths,
            "safety_violations": self.safety_violations,
        }
        
        results_path = os.path.join(self.config.output_dir, "final_results.pt")
        torch.save(results, results_path)
        
        # Save CSV results
        csv_path = os.path.join(self.config.output_dir, "training_results.csv")
        import pandas as pd
        
        df = pd.DataFrame({
            "episode": range(len(self.episode_rewards)),
            "reward": self.episode_rewards,
            "cost": self.episode_costs,
            "length": self.episode_lengths,
            "safety_violations": self.safety_violations,
        })
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved: {results_path}")
        print(f"CSV saved: {csv_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Safe RL Algorithm")
    parser.add_argument("--config", type=str, default="configs/safe_cpo.yaml", help="Config file path")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    config.output_dir = args.output_dir
    config.seed = args.seed
    
    # Create trainer and train
    trainer = SafeRLTrainer(config)
    results = trainer.train()
    
    print("Training completed!")
    print(f"Final results: {results}")


if __name__ == "__main__":
    main()
