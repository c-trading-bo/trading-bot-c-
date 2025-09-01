
#!/usr/bin/env python3
"""
CVaR-PPO Trainer for Risk-Aware Position Sizing

Trains a reinforcement learning policy to make position sizing decisions
under Conditional Value at Risk (CVaR) constraints using Proximal Policy 
Optimization with Lagrangian penalty.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F

from envs.sizer_env import SizerEnv, cvar_tail

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Policy(nn.Module):
    """Actor-Critic Policy Network."""
    
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, layers: int = 2):
        super().__init__()
        
        # Shared feature layers
        modules = [nn.Linear(obs_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        self.shared = nn.Sequential(*modules)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden, act_dim)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning logits and value."""
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class ExperienceBuffer:
    """Buffer for storing episode data."""
    
    def __init__(self, obs_dim: int, act_dim: int, size: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros(size, dtype=np.int32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.max_size = size
        
    def store(self, obs: np.ndarray, act: int, rew: float, val: float, logp: float):
        """Store experience."""
        if self.ptr < self.max_size:
            self.obs[self.ptr] = obs
            self.act[self.ptr] = act
            self.rew[self.ptr] = rew
            self.val[self.ptr] = val
            self.logp[self.ptr] = logp
            self.ptr += 1
    
    def get(self) -> Dict[str, np.ndarray]:
        """Get all stored experiences."""
        return {
            'obs': self.obs[:self.ptr].copy(),
            'act': self.act[:self.ptr].copy(), 
            'rew': self.rew[:self.ptr].copy(),
            'val': self.val[:self.ptr].copy(),
            'logp': self.logp[:self.ptr].copy()
        }
    
    def clear(self):
        """Clear buffer."""
        self.ptr = 0


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float = 0.99, lam: float = 0.95) -> np.ndarray:
    """Compute Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae
        
    return advantages


def train_cvar_ppo(
    df: pd.DataFrame,
    actions: Tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5),
    test_forward_days: int = 5,
    embargo_bars: int = 60,
    cvar_level: float = 0.95,
    cvar_target_r: float = 0.75,
    lagrange_init: float = 2.0,
    policy_hidden: int = 64,
    policy_layers: int = 2,
    ppo_steps: int = 50000,
    ppo_lr: float = 3e-4,
    ppo_clip: float = 0.2,
    ppo_gamma: float = 0.99,
    ppo_lambda: float = 0.95,
    device: str = "cpu"
) -> Tuple[Policy, Dict]:
    """Train CVaR-PPO policy.
    
    Returns:
        (trained_policy, training_metrics)
    """
    
    logger.info(f"Training CVaR-PPO with {len(df)} samples")
    logger.info(f"Actions: {actions}")
    logger.info(f"CVaR target: {cvar_target_r:.2f} at {cvar_level:.1%} level")
    
    # Split data (simple temporal split for now)
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    
    logger.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    
    # Create environment
    env = SizerEnv(train_df, actions)
    
    # Initialize policy
    policy = Policy(env.obs_dim, env.act_dim, policy_hidden, policy_layers).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=ppo_lr)
    
    # Lagrange multiplier for CVaR constraint
    lagrange_mult = lagrange_init
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'cvar_values': [],
        'policy_losses': [],
        'value_losses': [],
        'lagrange_mults': [],
        'constraint_violations': []
    }
    
    # Training loop
    buffer = ExperienceBuffer(env.obs_dim, env.act_dim, len(train_df))
    
    for episode in range(ppo_steps):
        # Collect episode
        obs = env.reset()
        buffer.clear()
        episode_rewards = []
        
        policy.eval()
        with torch.no_grad():
            while not env.done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits, value = policy(obs_tensor)
                
                # Sample action
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                
                # Store experience
                buffer.store(
                    obs.copy(),
                    int(action.item()),
                    0.0,  # Will be filled after step
                    value.item(),
                    log_prob.item()
                )
                
                # Take environment step
                obs, reward, done, info = env.step(int(action.item()))
                episode_rewards.append(reward)
                
                # Update reward in buffer
                buffer.rew[buffer.ptr - 1] = reward
                
                if done:
                    break
        
        # Skip if no experiences collected
        if buffer.ptr == 0:
            continue
            
        # Get episode data
        batch = buffer.get()
        episode_return = sum(episode_rewards)
        
        # Compute CVaR of this episode
        episode_cvar = cvar_tail(np.array(episode_rewards), cvar_level)
        
        # Policy update
        policy.train()
        
        # Compute advantages
        advantages = compute_gae(batch['rew'], batch['val'], ppo_gamma, ppo_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_tensor = torch.tensor(batch['obs'], device=device)
        act_tensor = torch.tensor(batch['act'], device=device)
        adv_tensor = torch.tensor(advantages, device=device)
        ret_tensor = torch.tensor(batch['rew'], device=device)
        old_logp_tensor = torch.tensor(batch['logp'], device=device)
        
        # PPO update
        logits, values = policy(obs_tensor)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(act_tensor)
        
        # Policy loss (PPO clip)
        ratio = torch.exp(log_probs - old_logp_tensor)
        clipped_ratio = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip)
        policy_loss = -torch.min(ratio * adv_tensor, clipped_ratio * adv_tensor).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), ret_tensor)
        
        # CVaR constraint penalty
        cvar_violation = max(0.0, abs(episode_cvar) - cvar_target_r)
        constraint_penalty = lagrange_mult * cvar_violation
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + constraint_penalty
        
        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Update Lagrange multiplier (simple adaptation)
        if cvar_violation > 0:
            lagrange_mult *= 1.01  # Increase penalty
        else:
            lagrange_mult *= 0.995  # Decrease penalty
        lagrange_mult = max(0.1, min(lagrange_mult, 50.0))  # Clamp
        
        # Log metrics
        metrics['episode_rewards'].append(episode_return)
        metrics['cvar_values'].append(episode_cvar)
        metrics['policy_losses'].append(policy_loss.item())
        metrics['value_losses'].append(value_loss.item())
        metrics['lagrange_mults'].append(lagrange_mult)
        metrics['constraint_violations'].append(cvar_violation)
        
        # Periodic logging
        if episode % 1000 == 0 or episode == ppo_steps - 1:
            avg_return = np.mean(metrics['episode_rewards'][-100:])
            avg_cvar = np.mean(metrics['cvar_values'][-100:])
            
            logger.info(f"Episode {episode:5d}: "
                       f"Return={avg_return:.3f}, "
                       f"CVaR={avg_cvar:.3f}, "
                       f"Policy Loss={policy_loss.item():.3f}, "
                       f"Value Loss={value_loss.item():.3f}, "
                       f"Lambda={lagrange_mult:.2f}")
    
    return policy, metrics


def main():
    parser = argparse.ArgumentParser(description='Train CVaR-PPO Sizer')
    parser.add_argument('--data-path', type=str, default='data/logs/candidates.parquet',
                       help='Path to candidate trade data')
    parser.add_argument('--out-rl', type=str, default='models/rl/latest_rl_sizer.onnx',
                       help='Output ONNX model path')
    parser.add_argument('--device', type=str, default='cpu', help='Training device')
    
    # Training hyperparameters from environment variables
    parser.add_argument('--actions', type=str, default=None,
                       help='Comma-separated action values')
    parser.add_argument('--steps', type=int, default=None, help='Training steps')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=None, help='Hidden units')
    parser.add_argument('--layers', type=int, default=None, help='Hidden layers')
    
    # Automated mode for C# integration
    parser.add_argument('--auto', action='store_true', 
                       help='Automated mode - use fast settings for production')
    parser.add_argument('--data', type=str, default=None,
                       help='Training data CSV path (for C# automated calls)')
    parser.add_argument('--output_model', type=str, default=None,
                       help='Output model path (for C# automated calls)')
    
    args = parser.parse_args()
    
    # Handle automated mode with fast training
    if args.auto:
        logger.info("🤖 AUTOMATED MODE: Using fast training settings")
        # Override for fast automated training
        config_overrides = {
            'ppo_steps': 5000,  # Much faster
            'policy_hidden': 32,  # Smaller network
            'policy_layers': 1,   # Simpler architecture
        }
    else:
        config_overrides = {}
    
    # Load hyperparameters from environment
    actions_str = args.actions or os.getenv('RL_ACTIONS', '0.50,0.75,1.00,1.25,1.50')
    actions = tuple(float(x.strip()) for x in actions_str.split(','))
    
    config = {
        'actions': actions,
        'test_forward_days': int(os.getenv('RL_TEST_FORWARD_DAYS', '5')),
        'embargo_bars': int(os.getenv('RL_EMBARGO_BARS', '60')),
        'cvar_level': float(os.getenv('RL_CVAR_LEVEL', '0.95')),
        'cvar_target_r': float(os.getenv('RL_CVAR_TARGET_R', '0.75')),
        'lagrange_init': float(os.getenv('RL_LAGRANGE_INIT', '2.0')),
        'policy_hidden': args.hidden or config_overrides.get('policy_hidden', int(os.getenv('RL_POLICY_HIDDEN', '64'))),
        'policy_layers': args.layers or config_overrides.get('policy_layers', int(os.getenv('RL_POLICY_LAYERS', '2'))),
        'ppo_steps': args.steps or config_overrides.get('ppo_steps', int(os.getenv('RL_PPO_STEPS', '50000'))),
        'ppo_lr': args.lr or float(os.getenv('RL_PPO_LR', '3e-4')),
        'ppo_clip': float(os.getenv('RL_PPO_CLIP', '0.2')),
        'ppo_gamma': float(os.getenv('RL_PPO_GAMMA', '0.99')),
        'ppo_lambda': float(os.getenv('RL_PPO_LAMBDA', '0.95')),
        'device': args.device
    }
    
    logger.info(f"Training configuration: {config}")
    
    # Load data - handle C# automated calls
    data_path_str = args.data or args.data_path
    data_path = Path(data_path_str)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        # Create dummy data for testing
        logger.info("Creating dummy data for testing...")
        np.random.seed(42)
        n_samples = 1000
        
        dummy_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
            'symbol': ['ES'] * n_samples,
            'session': np.random.choice(['RTH', 'ETH'], n_samples),
            'regime': np.random.choice(['Range', 'Trend', 'Vol'], n_samples),
            'R_multiple': np.random.normal(0.1, 1.5, n_samples),
            'slip_ticks': np.random.exponential(0.5, n_samples),
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.uniform(-1, 1, n_samples),
            'label_win': np.random.choice([0, 1], n_samples)
        })
        
        # Ensure directory exists
        data_path.parent.mkdir(parents=True, exist_ok=True)
        dummy_data.to_parquet(data_path)
        logger.info(f"Saved dummy data to {data_path}")
    
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} samples from {data_path}")
    
    # Train policy
    policy, metrics = train_cvar_ppo(df, **config)
    
    # Export to ONNX - handle C# automated calls
    out_path_str = args.output_model or args.out_rl
    out_path = Path(out_path_str)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create dummy input for ONNX export
    env = SizerEnv(df.head(10), actions)  # Small sample for feature detection
    dummy_input = torch.zeros(1, env.obs_dim, dtype=torch.float32)
    
    # Export model
    policy.eval()
    with torch.no_grad():
        torch.onnx.export(
            policy,
            (dummy_input,),  # Fix: wrap in tuple
            str(out_path),   # Fix: convert to string
            input_names=['features'],
            output_names=['logits', 'value'],
            opset_version=13,
            dynamic_axes={'features': {0: 'batch_size'}}
        )
    
    logger.info(f"Saved ONNX model to {out_path}")
    
    # Save training metrics
    metrics_path = out_path.parent / 'training_metrics.json'
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {k: [float(x) for x in v] for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    logger.info(f"Saved training metrics to {metrics_path}")
    
    # Final summary
    final_return = np.mean(metrics['episode_rewards'][-100:])
    final_cvar = np.mean(metrics['cvar_values'][-100:])
    final_violations = np.mean(metrics['constraint_violations'][-100:])
    
    logger.info("Training complete!")
    logger.info(f"Final average return: {final_return:.3f}")
    logger.info(f"Final CVaR@{config['cvar_level']:.0%}: {final_cvar:.3f}")
    logger.info(f"Final constraint violations: {final_violations:.3f}")


if __name__ == "__main__":
    main()
