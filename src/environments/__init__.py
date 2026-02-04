"""Safe reinforcement learning environments and wrappers."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Tuple, Optional, Union


class SafeCartPoleWrapper(gym.Wrapper):
    """Wrapper for CartPole with safety constraints.
    
    Adds safety constraints to the CartPole environment:
    - Position constraint: |cart_position| <= safety_threshold
    - Velocity constraint: |cart_velocity| <= velocity_threshold
    """
    
    def __init__(
        self,
        env: gym.Env,
        safety_threshold: float = 2.0,
        velocity_threshold: float = 3.0,
        safety_penalty: float = -10.0,
    ):
        """Initialize safe CartPole wrapper.
        
        Args:
            env: Base CartPole environment
            safety_threshold: Maximum allowed cart position
            velocity_threshold: Maximum allowed cart velocity
            safety_penalty: Penalty for safety violations
        """
        super().__init__(env)
        self.safety_threshold = safety_threshold
        self.velocity_threshold = velocity_threshold
        self.safety_penalty = safety_penalty
        
        # Add safety info to observation space
        self.observation_space = spaces.Box(
            low=np.concatenate([env.observation_space.low, [-1.0]]),
            high=np.concatenate([env.observation_space.high, [1.0]]),
            dtype=np.float32,
        )
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment with safety monitoring.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check safety constraints
        cart_pos, cart_vel = obs[0], obs[2]
        safety_violation = (
            abs(cart_pos) > self.safety_threshold or 
            abs(cart_vel) > self.velocity_threshold
        )
        
        # Apply safety penalty
        if safety_violation:
            reward += self.safety_penalty
        
        # Add safety information
        safety_info = {
            "safety_violation": safety_violation,
            "cart_position": cart_pos,
            "cart_velocity": cart_vel,
            "position_violation": abs(cart_pos) > self.safety_threshold,
            "velocity_violation": abs(cart_vel) > self.velocity_threshold,
        }
        info.update(safety_info)
        
        # Add safety flag to observation
        safety_flag = 1.0 if safety_violation else -1.0
        obs_with_safety = np.concatenate([obs, [safety_flag]])
        
        return obs_with_safety, reward, terminated, truncated, info
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)
        safety_flag = -1.0  # No violation at start
        obs_with_safety = np.concatenate([obs, [safety_flag]])
        return obs_with_safety, info


class ConstrainedMountainCarWrapper(gym.Wrapper):
    """Wrapper for MountainCar with energy constraints.
    
    Adds energy constraints to prevent excessive exploration.
    """
    
    def __init__(
        self,
        env: gym.Env,
        max_energy: float = 1000.0,
        energy_penalty: float = -1.0,
    ):
        """Initialize constrained MountainCar wrapper.
        
        Args:
            env: Base MountainCar environment
            max_energy: Maximum allowed energy consumption
            energy_penalty: Penalty for energy violations
        """
        super().__init__(env)
        self.max_energy = max_energy
        self.energy_penalty = energy_penalty
        self.energy_consumed = 0.0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment with energy monitoring.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Track energy consumption (simplified as action magnitude)
        self.energy_consumed += abs(action)
        
        # Check energy constraint
        energy_violation = self.energy_consumed > self.max_energy
        
        # Apply energy penalty
        if energy_violation:
            reward += self.energy_penalty
        
        # Add energy information
        energy_info = {
            "energy_consumed": self.energy_consumed,
            "energy_violation": energy_violation,
            "energy_remaining": max(0, self.max_energy - self.energy_consumed),
        }
        info.update(energy_info)
        
        return obs, reward, terminated, truncated, info
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.energy_consumed = 0.0
        return obs, info


class SafetyMonitorWrapper(gym.Wrapper):
    """Generic wrapper for monitoring safety metrics."""
    
    def __init__(self, env: gym.Env):
        """Initialize safety monitor wrapper.
        
        Args:
            env: Base environment
        """
        super().__init__(env)
        self.safety_violations = 0
        self.total_steps = 0
        self.episode_violations = 0
        self.episode_steps = 0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment with safety monitoring.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.total_steps += 1
        self.episode_steps += 1
        
        # Check for safety violations in info
        if "safety_violation" in info and info["safety_violation"]:
            self.safety_violations += 1
            self.episode_violations += 1
        
        # Add cumulative safety metrics
        safety_metrics = {
            "total_safety_violations": self.safety_violations,
            "total_steps": self.total_steps,
            "episode_safety_violations": self.episode_violations,
            "episode_steps": self.episode_steps,
            "safety_violation_rate": (
                self.safety_violations / self.total_steps if self.total_steps > 0 else 0
            ),
        }
        info.update(safety_metrics)
        
        # Reset episode metrics on episode end
        if terminated or truncated:
            self.episode_violations = 0
            self.episode_steps = 0
        
        return obs, reward, terminated, truncated, info
    
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            Tuple of (observation, info)
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self.episode_violations = 0
        self.episode_steps = 0
        return obs, info


def make_safe_env(
    env_name: str,
    safety_threshold: float = 2.0,
    velocity_threshold: float = 3.0,
    safety_penalty: float = -10.0,
    **kwargs
) -> gym.Env:
    """Create a safe environment with appropriate wrappers.
    
    Args:
        env_name: Name of the environment
        safety_threshold: Safety threshold for position constraints
        velocity_threshold: Safety threshold for velocity constraints
        safety_penalty: Penalty for safety violations
        **kwargs: Additional environment arguments
        
    Returns:
        Wrapped safe environment
    """
    env = gym.make(env_name, **kwargs)
    
    if "CartPole" in env_name:
        env = SafeCartPoleWrapper(
            env,
            safety_threshold=safety_threshold,
            velocity_threshold=velocity_threshold,
            safety_penalty=safety_penalty,
        )
    elif "MountainCar" in env_name:
        env = ConstrainedMountainCarWrapper(env)
    
    env = SafetyMonitorWrapper(env)
    return env
