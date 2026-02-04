"""Comprehensive evaluation script for safe RL algorithms."""

import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from src.algorithms import SafeQLearning, ConstrainedPolicyOptimization, LagrangianSafeRL
from src.environments import make_safe_env
from src.evaluation import SafeRLEvaluator
from src.utils import set_seed, get_device


def evaluate_algorithm(
    algorithm_name: str,
    config: DictConfig,
    num_runs: int = 5,
    num_episodes: int = 1000,
    num_eval_episodes: int = 100,
) -> Dict[str, any]:
    """Evaluate a single algorithm across multiple runs.
    
    Args:
        algorithm_name: Name of the algorithm
        config: Configuration object
        num_runs: Number of independent runs
        num_episodes: Number of training episodes per run
        num_eval_episodes: Number of evaluation episodes per run
        
    Returns:
        Evaluation results
    """
    print(f"Evaluating {algorithm_name}...")
    
    all_results = []
    
    for run in range(num_runs):
        print(f"  Run {run + 1}/{num_runs}")
        
        # Set seed for this run
        run_seed = config.seed + run * 1000
        set_seed(run_seed)
        
        # Create environment
        env = make_safe_env(
            config.env.name,
            safety_threshold=config.env.safety_threshold,
            velocity_threshold=config.env.velocity_threshold,
            safety_penalty=config.env.safety_penalty,
        )
        
        # Create algorithm
        device = get_device()
        if algorithm_name == "SafeQLearning":
            algorithm = SafeQLearning(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                learning_rate=config.algorithm.learning_rate,
                gamma=config.algorithm.gamma,
                epsilon=config.algorithm.epsilon,
                epsilon_decay=config.algorithm.epsilon_decay,
                epsilon_min=config.algorithm.epsilon_min,
                safety_threshold=config.env.safety_threshold,
                safety_penalty=config.env.safety_penalty,
                device=device,
            )
        elif algorithm_name == "CPO":
            algorithm = ConstrainedPolicyOptimization(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                hidden_dims=tuple(config.algorithm.hidden_dims),
                learning_rate=config.algorithm.learning_rate,
                gamma=config.algorithm.gamma,
                lam=config.algorithm.lam,
                clip_ratio=config.algorithm.clip_ratio,
                value_loss_coef=config.algorithm.value_loss_coef,
                entropy_coef=config.algorithm.entropy_coef,
                max_grad_norm=config.algorithm.max_grad_norm,
                cost_limit=config.algorithm.cost_limit,
                cpo_iters=config.algorithm.cpo_iters,
                cpo_step_size=config.algorithm.cpo_step_size,
                device=device,
            )
        elif algorithm_name == "Lagrangian":
            algorithm = LagrangianSafeRL(
                state_dim=env.observation_space.shape[0],
                action_dim=env.action_space.n,
                hidden_dims=tuple(config.algorithm.hidden_dims),
                learning_rate=config.algorithm.learning_rate,
                lagrangian_lr=config.algorithm.lagrangian_lr,
                gamma=config.algorithm.gamma,
                lam=config.algorithm.lam,
                clip_ratio=config.algorithm.clip_ratio,
                value_loss_coef=config.algorithm.value_loss_coef,
                entropy_coef=config.algorithm.entropy_coef,
                max_grad_norm=config.algorithm.max_grad_norm,
                cost_limit=config.algorithm.cost_limit,
                device=device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Training
        training_rewards = []
        training_costs = []
        training_lengths = []
        training_safety_violations = []
        
        for episode in range(num_episodes):
            state, _ = env.reset(seed=run_seed + episode)
            done = False
            episode_reward = 0
            episode_cost = 0
            episode_length = 0
            episode_safety_violations = 0
            
            episode_rewards = []
            episode_costs = []
            episode_safety_violations_list = []
            
            while not done:
                if algorithm_name == "SafeQLearning":
                    action = algorithm.select_action(state, training=True)
                else:
                    action, _, _ = algorithm.select_action(state, training=True)
                
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                safety_violation = info.get("safety_violation", False)
                cost = 1.0 if safety_violation else 0.0
                
                episode_reward += reward
                episode_cost += cost
                episode_length += 1
                if safety_violation:
                    episode_safety_violations += 1
                
                episode_rewards.append(reward)
                episode_costs.append(cost)
                episode_safety_violations_list.append(safety_violation)
                
                state = next_state
            
            training_rewards.append(episode_reward)
            training_costs.append(episode_cost)
            training_lengths.append(episode_length)
            training_safety_violations.append(episode_safety_violations)
        
        # Evaluation
        eval_rewards = []
        eval_costs = []
        eval_lengths = []
        eval_safety_violations = []
        
        for eval_episode in range(num_eval_episodes):
            state, _ = env.reset(seed=run_seed + 10000 + eval_episode)
            done = False
            episode_reward = 0
            episode_cost = 0
            episode_length = 0
            episode_safety_violations = 0
            
            while not done:
                if algorithm_name == "SafeQLearning":
                    action = algorithm.select_action(state, training=False)
                else:
                    action, _, _ = algorithm.select_action(state, training=False)
                
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
        
        # Store run results
        run_results = {
            "algorithm": algorithm_name,
            "run": run,
            "seed": run_seed,
            "training_rewards": training_rewards,
            "training_costs": training_costs,
            "training_lengths": training_lengths,
            "training_safety_violations": training_safety_violations,
            "eval_rewards": eval_rewards,
            "eval_costs": eval_costs,
            "eval_lengths": eval_lengths,
            "eval_safety_violations": eval_safety_violations,
        }
        
        all_results.append(run_results)
    
    # Aggregate results across runs
    eval_rewards_all = []
    eval_costs_all = []
    eval_lengths_all = []
    eval_safety_violations_all = []
    
    for result in all_results:
        eval_rewards_all.extend(result["eval_rewards"])
        eval_costs_all.extend(result["eval_costs"])
        eval_lengths_all.extend(result["eval_lengths"])
        eval_safety_violations_all.extend(result["eval_safety_violations"])
    
    # Compute final metrics
    final_results = {
        "algorithm": algorithm_name,
        "num_runs": num_runs,
        "num_episodes": num_episodes,
        "num_eval_episodes": num_eval_episodes,
        "mean_reward": np.mean(eval_rewards_all),
        "std_reward": np.std(eval_rewards_all),
        "mean_cost": np.mean(eval_costs_all),
        "std_cost": np.std(eval_costs_all),
        "mean_length": np.mean(eval_lengths_all),
        "std_length": np.std(eval_lengths_all),
        "mean_safety_violations": np.mean(eval_safety_violations_all),
        "std_safety_violations": np.std(eval_safety_violations_all),
        "safety_violation_rate": np.mean(eval_safety_violations_all) / np.mean(eval_lengths_all),
        "constraint_satisfaction_rate": np.mean([c <= config.algorithm.cost_limit for c in eval_costs_all]),
        "all_results": all_results,
    }
    
    print(f"  Mean Reward: {final_results['mean_reward']:.2f} ± {final_results['std_reward']:.2f}")
    print(f"  Mean Cost: {final_results['mean_cost']:.2f} ± {final_results['std_cost']:.2f}")
    print(f"  Constraint Satisfaction Rate: {final_results['constraint_satisfaction_rate']:.3f}")
    print()
    
    return final_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Comprehensive Safe RL Evaluation")
    parser.add_argument("--config", type=str, default="configs/safe_cpo.yaml", help="Config file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--num-episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--num-eval-episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--algorithms", nargs="+", default=["SafeQLearning", "CPO", "Lagrangian"], help="Algorithms to evaluate")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Safe Reinforcement Learning Comprehensive Evaluation")
    print("=" * 60)
    print(f"Environment: {config.env.name}")
    print(f"Safety Threshold: {config.env.safety_threshold}")
    print(f"Safety Penalty: {config.env.safety_penalty}")
    print(f"Cost Limit: {config.algorithm.cost_limit}")
    print(f"Number of Runs: {args.num_runs}")
    print(f"Training Episodes: {args.num_episodes}")
    print(f"Evaluation Episodes: {args.num_eval_episodes}")
    print(f"Algorithms: {args.algorithms}")
    print()
    
    # Evaluate each algorithm
    all_results = []
    
    for algorithm in args.algorithms:
        start_time = time.time()
        results = evaluate_algorithm(
            algorithm, config, args.num_runs, args.num_episodes, args.num_eval_episodes
        )
        evaluation_time = time.time() - start_time
        results["evaluation_time"] = evaluation_time
        all_results.append(results)
    
    # Create comparison table
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    comparison_data = []
    for result in all_results:
        comparison_data.append({
            "Algorithm": result["algorithm"],
            "Mean Reward": f"{result['mean_reward']:.2f} ± {result['std_reward']:.2f}",
            "Mean Cost": f"{result['mean_cost']:.2f} ± {result['std_cost']:.2f}",
            "Mean Length": f"{result['mean_length']:.2f} ± {result['std_length']:.2f}",
            "Safety Violation Rate": f"{result['safety_violation_rate']:.3f}",
            "Constraint Satisfaction": f"{result['constraint_satisfaction_rate']:.3f}",
            "Evaluation Time": f"{result['evaluation_time']:.1f}s",
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save results
    results_path = os.path.join(args.output_dir, "comprehensive_evaluation_results.pt")
    torch.save(all_results, results_path)
    
    csv_path = os.path.join(args.output_dir, "comparison_table.csv")
    comparison_df.to_csv(csv_path, index=False)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Comparison table saved to: {csv_path}")
    
    # Safety disclaimer
    print("\n" + "=" * 60)
    print("SAFETY DISCLAIMER")
    print("=" * 60)
    print("This evaluation is for research and educational purposes only.")
    print("The algorithms evaluated should NOT be used for controlling")
    print("real-world systems, especially in safety-critical domains.")


if __name__ == "__main__":
    main()
