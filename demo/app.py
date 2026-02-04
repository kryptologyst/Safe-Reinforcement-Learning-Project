"""Streamlit demo for Safe Reinforcement Learning."""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from algorithms import ConstrainedPolicyOptimization, LagrangianSafeRL, SafeQLearning
from environments import make_safe_env
from evaluation import SafeRLEvaluator
from utils import set_seed, get_device


# Page configuration
st.set_page_config(
    page_title="Safe RL Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("🛡️ Safe Reinforcement Learning Demo")
st.markdown("""
**WARNING: This is a research/educational demo only. NOT for production control of real systems.**

This demo showcases safe reinforcement learning algorithms with constraint satisfaction and safety monitoring.
""")

# Sidebar configuration
st.sidebar.header("Configuration")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["Safe Q-Learning", "Constrained Policy Optimization (CPO)", "Lagrangian Safe RL"],
    index=1
)

# Environment selection
env_name = st.sidebar.selectbox(
    "Environment",
    ["CartPole-v1", "MountainCar-v0"],
    index=0
)

# Safety parameters
st.sidebar.subheader("Safety Parameters")
safety_threshold = st.sidebar.slider("Safety Threshold", 1.0, 5.0, 2.0, 0.1)
velocity_threshold = st.sidebar.slider("Velocity Threshold", 1.0, 10.0, 3.0, 0.1)
safety_penalty = st.sidebar.slider("Safety Penalty", -50.0, 0.0, -10.0, 1.0)

# Training parameters
st.sidebar.subheader("Training Parameters")
num_episodes = st.sidebar.slider("Number of Episodes", 100, 2000, 500, 50)
learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 3e-4, 1e-5, format="%.2e")
cost_limit = st.sidebar.slider("Cost Limit", 0.001, 0.1, 0.01, 0.001)

# Evaluation parameters
st.sidebar.subheader("Evaluation Parameters")
num_eval_episodes = st.sidebar.slider("Evaluation Episodes", 10, 100, 50, 5)
seed = st.sidebar.number_input("Random Seed", 1, 10000, 42)

# Initialize session state
if "training_results" not in st.session_state:
    st.session_state.training_results = None
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "episode_data" not in st.session_state:
    st.session_state.episode_data = []


def create_algorithm(algorithm_name: str, state_dim: int, action_dim: int, device: torch.device):
    """Create algorithm instance."""
    if algorithm_name == "Safe Q-Learning":
        return SafeQLearning(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            safety_threshold=safety_threshold,
            safety_penalty=safety_penalty,
            device=device,
        )
    elif algorithm_name == "Constrained Policy Optimization (CPO)":
        return ConstrainedPolicyOptimization(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            cost_limit=cost_limit,
            device=device,
        )
    elif algorithm_name == "Lagrangian Safe RL":
        return LagrangianSafeRL(
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            cost_limit=cost_limit,
            device=device,
        )


def train_algorithm():
    """Train the selected algorithm."""
    st.info("Starting training...")
    
    # Set random seed
    set_seed(seed)
    
    # Create environment
    env = make_safe_env(
        env_name,
        safety_threshold=safety_threshold,
        velocity_threshold=velocity_threshold,
        safety_penalty=safety_penalty,
    )
    
    # Create algorithm
    device = get_device()
    algorithm = create_algorithm(algorithm, env.observation_space.shape[0], env.action_space.n, device)
    
    # Training progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Training loop
    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    safety_violations = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0
        episode_cost = 0
        episode_length = 0
        episode_safety_violations = 0
        
        while not done:
            # Select action
            if algorithm_name == "Safe Q-Learning":
                action = algorithm.select_action(state, training=True)
            else:
                action, _, _ = algorithm.select_action(state, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract safety information
            cost = info.get("safety_violation", False) * 1.0
            safety_violation = info.get("safety_violation", False)
            
            episode_reward += reward
            episode_cost += cost
            episode_length += 1
            if safety_violation:
                episode_safety_violations += 1
            
            state = next_state
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_lengths.append(episode_length)
        safety_violations.append(episode_safety_violations)
        
        # Update progress
        progress = (episode + 1) / num_episodes
        progress_bar.progress(progress)
        status_text.text(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}, Cost: {episode_cost:.2f}")
    
    # Store results
    st.session_state.training_results = {
        "algorithm": algorithm,
        "env": env,
        "episode_rewards": episode_rewards,
        "episode_costs": episode_costs,
        "episode_lengths": episode_lengths,
        "safety_violations": safety_violations,
    }
    
    st.success("Training completed!")
    return st.session_state.training_results


def evaluate_algorithm():
    """Evaluate the trained algorithm."""
    if st.session_state.training_results is None:
        st.error("Please train an algorithm first!")
        return
    
    st.info("Evaluating algorithm...")
    
    algorithm = st.session_state.training_results["algorithm"]
    env = st.session_state.training_results["env"]
    
    # Evaluation progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    eval_rewards = []
    eval_costs = []
    eval_lengths = []
    eval_safety_violations = []
    
    for episode in range(num_eval_episodes):
        state, _ = env.reset(seed=seed + 10000 + episode)
        done = False
        episode_reward = 0
        episode_cost = 0
        episode_length = 0
        episode_safety_violations = 0
        
        while not done:
            # Select action (no exploration)
            if algorithm_name == "Safe Q-Learning":
                action = algorithm.select_action(state, training=False)
            else:
                action, _, _ = algorithm.select_action(state, training=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Extract safety information
            cost = info.get("safety_violation", False) * 1.0
            safety_violation = info.get("safety_violation", False)
            
            episode_reward += reward
            episode_cost += cost
            episode_length += 1
            if safety_violation:
                episode_safety_violations += 1
            
            state = next_state
        
        # Store episode results
        eval_rewards.append(episode_reward)
        eval_costs.append(episode_cost)
        eval_lengths.append(episode_length)
        eval_safety_violations.append(episode_safety_violations)
        
        # Update progress
        progress = (episode + 1) / num_eval_episodes
        progress_bar.progress(progress)
        status_text.text(f"Evaluation Episode {episode + 1}/{num_eval_episodes}")
    
    # Store evaluation results
    st.session_state.evaluation_results = {
        "rewards": eval_rewards,
        "costs": eval_costs,
        "lengths": eval_lengths,
        "safety_violations": eval_safety_violations,
    }
    
    st.success("Evaluation completed!")


def plot_training_results():
    """Plot training results."""
    if st.session_state.training_results is None:
        st.error("Please train an algorithm first!")
        return
    
    results = st.session_state.training_results
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Episode Rewards", "Episode Costs", "Episode Lengths", "Safety Violations"),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )
    
    episodes = list(range(len(results["episode_rewards"])))
    
    # Plot rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=results["episode_rewards"], mode="lines", name="Rewards"),
        row=1, col=1
    )
    
    # Plot costs
    fig.add_trace(
        go.Scatter(x=episodes, y=results["episode_costs"], mode="lines", name="Costs"),
        row=1, col=2
    )
    
    # Plot lengths
    fig.add_trace(
        go.Scatter(x=episodes, y=results["episode_lengths"], mode="lines", name="Lengths"),
        row=2, col=1
    )
    
    # Plot safety violations
    fig.add_trace(
        go.Scatter(x=episodes, y=results["safety_violations"], mode="lines", name="Safety Violations"),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Training Results",
        height=600,
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_evaluation_results():
    """Plot evaluation results."""
    if st.session_state.evaluation_results is None:
        st.error("Please evaluate the algorithm first!")
        return
    
    results = st.session_state.evaluation_results
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Reward Distribution", "Cost Distribution", "Length Distribution", "Safety Violations"),
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )
    
    # Plot reward distribution
    fig.add_trace(
        go.Histogram(x=results["rewards"], name="Rewards", nbinsx=20),
        row=1, col=1
    )
    
    # Plot cost distribution
    fig.add_trace(
        go.Histogram(x=results["costs"], name="Costs", nbinsx=20),
        row=1, col=2
    )
    
    # Plot length distribution
    fig.add_trace(
        go.Histogram(x=results["lengths"], name="Lengths", nbinsx=20),
        row=2, col=1
    )
    
    # Plot safety violations
    fig.add_trace(
        go.Histogram(x=results["safety_violations"], name="Safety Violations", nbinsx=20),
        row=2, col=2
    )
    
    fig.update_layout(
        title="Evaluation Results",
        height=600,
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_evaluation_metrics():
    """Show evaluation metrics."""
    if st.session_state.evaluation_results is None:
        st.error("Please evaluate the algorithm first!")
        return
    
    results = st.session_state.evaluation_results
    
    # Compute metrics
    metrics = {
        "Mean Reward": np.mean(results["rewards"]),
        "Std Reward": np.std(results["rewards"]),
        "Mean Cost": np.mean(results["costs"]),
        "Std Cost": np.std(results["costs"]),
        "Mean Length": np.mean(results["lengths"]),
        "Std Length": np.std(results["lengths"]),
        "Mean Safety Violations": np.mean(results["safety_violations"]),
        "Std Safety Violations": np.std(results["safety_violations"]),
        "Safety Violation Rate": np.mean(results["safety_violations"]) / np.mean(results["lengths"]),
        "Constraint Satisfaction Rate": np.mean([c <= cost_limit for c in results["costs"]]),
    }
    
    # Display metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        for key, value in list(metrics.items())[:5]:
            st.metric(key, f"{value:.3f}")
    
    with col2:
        st.subheader("Safety Metrics")
        for key, value in list(metrics.items())[5:]:
            st.metric(key, f"{value:.3f}")


# Main interface
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🚀 Train Algorithm", type="primary"):
        train_algorithm()

with col2:
    if st.button("📊 Evaluate Algorithm"):
        evaluate_algorithm()

with col3:
    if st.button("🔄 Reset"):
        st.session_state.training_results = None
        st.session_state.evaluation_results = None
        st.session_state.episode_data = []
        st.rerun()

# Display results
if st.session_state.training_results is not None:
    st.header("Training Results")
    plot_training_results()

if st.session_state.evaluation_results is not None:
    st.header("Evaluation Results")
    plot_evaluation_results()
    show_evaluation_metrics()

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This demo is for research and educational purposes only. 
The algorithms shown here should NOT be used for controlling real-world systems, 
especially in safety-critical domains such as autonomous vehicles, medical devices, 
or industrial control systems.
""")
