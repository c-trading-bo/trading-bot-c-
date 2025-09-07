#!/usr/bin/env python3
"""
Proximal Policy Optimization (PPO) Training Script for Trading Bot

This script trains a PPO agent for dynamic trading decisions.
PPO is used for action selection and policy optimization in the trading environment.

Integration with C# system:
- Exports ONNX models for C# integration
- Compatible with existing ML pipeline
- Supports both discrete and continuous action spaces
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from datetime import datetime, timedelta
import json
import argparse
from pathlib import Path
import pandas as pd
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class PPOActor(nn.Module):
    """
    PPO Actor Network for policy-based trading decisions.
    
    Outputs action probabilities for:
    - Strategy selection (discrete)
    - Position sizing (continuous) 
    - Entry/exit timing (continuous)
    """
    
    def __init__(self, state_dim=20, action_dim=14, hidden_dims=[128, 64]):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Strategy selection head (discrete actions)
        self.strategy_head = nn.Linear(prev_dim, action_dim)
        
        # Position sizing head (continuous actions)
        self.position_mean = nn.Linear(prev_dim, 1)
        self.position_std = nn.Linear(prev_dim, 1)
        
        # Value function head
        self.value_head = nn.Linear(prev_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using orthogonal initialization for RL."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action_mask=None):
        """
        Forward pass returning policy and value.
        
        Args:
            state: Market state features [batch_size, state_dim]
            action_mask: Optional mask for invalid actions [batch_size, action_dim]
        
        Returns:
            dict with strategy_logits, position_params, value
        """
        features = self.shared_layers(state)
        
        # Strategy selection (discrete)
        strategy_logits = self.strategy_head(features)
        if action_mask is not None:
            strategy_logits = strategy_logits.masked_fill(action_mask == 0, -1e9)
        
        # Position sizing (continuous, bounded [0.1, 2.0])
        position_mean = torch.sigmoid(self.position_mean(features)) * 1.9 + 0.1  # [0.1, 2.0]
        position_std = torch.softplus(self.position_std(features)) + 1e-4  # Ensure positive
        
        # Value function
        value = self.value_head(features)
        
        return {
            'strategy_logits': strategy_logits,
            'position_mean': position_mean,
            'position_std': position_std,
            'value': value
        }
    
    def get_action_and_value(self, state, action_mask=None):
        """Sample actions and compute value for given state."""
        output = self.forward(state, action_mask)
        
        # Sample strategy
        strategy_dist = Categorical(logits=output['strategy_logits'])
        strategy_action = strategy_dist.sample()
        strategy_log_prob = strategy_dist.log_prob(strategy_action)
        
        # Sample position size
        position_dist = Normal(output['position_mean'], output['position_std'])
        position_action = position_dist.sample()
        position_log_prob = position_dist.log_prob(position_action)
        
        # Clamp position size to valid range
        position_action = torch.clamp(position_action, 0.1, 2.0)
        
        return {
            'strategy_action': strategy_action,
            'position_action': position_action,
            'strategy_log_prob': strategy_log_prob,
            'position_log_prob': position_log_prob,
            'value': output['value']
        }
    
    def evaluate_actions(self, state, strategy_action, position_action, action_mask=None):
        """Evaluate actions for PPO updates."""
        output = self.forward(state, action_mask)
        
        # Strategy evaluation
        strategy_dist = Categorical(logits=output['strategy_logits'])
        strategy_log_prob = strategy_dist.log_prob(strategy_action)
        strategy_entropy = strategy_dist.entropy()
        
        # Position evaluation
        position_dist = Normal(output['position_mean'], output['position_std'])
        position_log_prob = position_dist.log_prob(position_action)
        position_entropy = position_dist.entropy()
        
        return {
            'strategy_log_prob': strategy_log_prob,
            'position_log_prob': position_log_prob,
            'entropy': strategy_entropy + position_entropy,
            'value': output['value']
        }

class PPOTrainer:
    """
    PPO trainer with clipped policy updates and value function learning.
    
    Features:
    - Clipped surrogate objective
    - Value function training
    - Entropy regularization
    - Gradient clipping
    - Learning rate scheduling
    """
    
    def __init__(self, actor, learning_rate=3e-4, device='cpu'):
        self.actor = actor
        self.device = device
        self.actor.to(device)
        
        self.optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)
        
        # PPO hyperparameters
        self.clip_ratio = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
        }
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)
    
    def update_policy(self, states, strategy_actions, position_actions, old_strategy_log_probs, 
                     old_position_log_probs, advantages, returns, action_masks=None):
        """Perform PPO policy update."""
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        strategy_actions = torch.LongTensor(strategy_actions).to(self.device)
        position_actions = torch.FloatTensor(position_actions).to(self.device)
        old_strategy_log_probs = torch.FloatTensor(old_strategy_log_probs).to(self.device)
        old_position_log_probs = torch.FloatTensor(old_position_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs of updates
        for _ in range(4):  # PPO typically does 4 epochs
            # Evaluate current policy
            eval_output = self.actor.evaluate_actions(
                states, strategy_actions, position_actions, action_masks
            )
            
            # Compute policy ratios
            strategy_ratio = torch.exp(eval_output['strategy_log_prob'] - old_strategy_log_probs)
            position_ratio = torch.exp(eval_output['position_log_prob'] - old_position_log_probs)
            
            # Combined ratio
            ratio = strategy_ratio * position_ratio
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            value_loss = F.mse_loss(eval_output['value'].squeeze(), returns)
            
            # Entropy loss
            entropy_loss = -eval_output['entropy'].mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Store stats
            self.training_stats['policy_loss'].append(policy_loss.item())
            self.training_stats['value_loss'].append(value_loss.item())
            self.training_stats['entropy'].append(-entropy_loss.item())
            self.training_stats['total_loss'].append(total_loss.item())
        
        self.scheduler.step()

def create_trading_environment_data(n_episodes=1000):
    """
    Create synthetic trading environment data for PPO training.
    
    Simulates episodes of trading with states, actions, and rewards.
    """
    print(f"Generating {n_episodes} episodes of trading data...")
    
    episodes = []
    
    for episode in range(n_episodes):
        episode_length = np.random.randint(50, 200)  # Variable episode length
        
        states = []
        actions = []
        rewards = []
        dones = []
        
        # Initial market state
        price = 4500 + np.random.normal(0, 100)
        
        for step in range(episode_length):
            # Market state features (20 dimensions)
            state = np.array([
                price / 5000,  # Normalized price
                np.random.uniform(-0.1, 0.1),  # Price change
                np.random.uniform(0, 1),  # Volume
                np.random.uniform(0, 1),  # ATR
                np.random.uniform(0, 1),  # RSI
                np.random.uniform(0, 1),  # Bollinger position
                np.random.uniform(0, 1),  # MACD
                np.random.uniform(0, 1),  # Volatility
                np.random.uniform(0, 1),  # Spread
                step / episode_length,  # Time in episode
                np.sin(step * 2 * np.pi / 100),  # Cyclical time
                np.cos(step * 2 * np.pi / 100),
                np.random.uniform(0, 1),  # Market regime
                np.random.uniform(0, 1),  # News sentiment
                np.random.uniform(0, 1),  # Economic indicator
                np.random.uniform(0, 1),  # Option flow
                np.random.uniform(0, 1),  # Futures basis
                np.random.uniform(0, 1),  # Sector rotation
                np.random.uniform(0, 1),  # VIX level
                np.random.uniform(0, 1)   # Dollar strength
            ])
            
            states.append(state)
            
            # Action: strategy selection (0-13) and position size (0.1-2.0)
            strategy_action = np.random.randint(0, 14)
            position_action = np.random.uniform(0.1, 2.0)
            actions.append((strategy_action, position_action))
            
            # Reward based on strategy performance and market conditions
            base_reward = np.random.normal(0, 1)
            
            # Add some logic to make certain strategies better in certain conditions
            if state[4] > 0.7 and strategy_action in [2, 3]:  # High RSI + mean reversion strategies
                base_reward += 0.5
            elif state[1] > 0 and strategy_action in [1, 6]:  # Uptrend + momentum strategies
                base_reward += 0.3
            
            # Position size affects reward
            if position_action > 1.5:
                base_reward *= 0.8  # Penalty for oversizing
            
            rewards.append(base_reward)
            
            # Update price for next step
            price += np.random.normal(0, 20)
            price = max(4000, min(5000, price))  # Keep price in reasonable range
            
            # Episode termination
            done = step == episode_length - 1
            dones.append(done)
        
        episodes.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'dones': dones
        })
    
    return episodes

def train_ppo_agent(args):
    """
    Main training function for PPO agent.
    """
    print("=" * 60)
    print("PPO AGENT TRAINING")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    actor = PPOActor(state_dim=20, action_dim=14, hidden_dims=[128, 64])
    trainer = PPOTrainer(actor, learning_rate=3e-4, device=device)
    
    print(f"Created PPO agent with {sum(p.numel() for p in actor.parameters())} parameters")
    
    # Generate training data
    episodes = create_trading_environment_data(n_episodes=args.episodes)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        epoch_rewards = []
        
        # Process episodes in batches
        for episode in episodes:
            states = np.array(episode['states'])
            actions = episode['actions']
            rewards = episode['rewards']
            dones = episode['dones']
            
            # Split actions
            strategy_actions = [a[0] for a in actions]
            position_actions = [a[1] for a in actions]
            
            # Get old policy predictions
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states).to(device)
                old_output = actor.get_action_and_value(states_tensor)
                values = old_output['value'].cpu().numpy().flatten()
                old_strategy_log_probs = old_output['strategy_log_prob'].cpu().numpy()
                old_position_log_probs = old_output['position_log_prob'].cpu().numpy()
            
            # Compute advantages and returns
            advantages, returns = trainer.compute_gae(rewards, values, dones)
            
            # Update policy
            trainer.update_policy(
                states, strategy_actions, position_actions,
                old_strategy_log_probs, old_position_log_probs,
                advantages.numpy(), returns.numpy()
            )
            
            epoch_rewards.append(sum(rewards))
        
        # Log progress
        if epoch % 10 == 0:
            avg_reward = np.mean(epoch_rewards)
            avg_policy_loss = np.mean(trainer.training_stats['policy_loss'][-100:])
            avg_value_loss = np.mean(trainer.training_stats['value_loss'][-100:])
            
            print(f"Epoch {epoch:3d}: Avg Reward = {avg_reward:8.3f}, "
                  f"Policy Loss = {avg_policy_loss:.6f}, Value Loss = {avg_value_loss:.6f}")
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save PyTorch model
    torch_path = output_dir / f"ppo_agent_{timestamp}.pth"
    torch.save({
        'model_state_dict': actor.state_dict(),
        'model_config': {
            'state_dim': 20,
            'action_dim': 14,
            'hidden_dims': [128, 64]
        },
        'training_stats': trainer.training_stats
    }, torch_path)
    
    # Export ONNX model
    onnx_path = output_dir / f"ppo_agent_{timestamp}.onnx"
    actor.eval()
    dummy_input = torch.randn(1, 20).to(device)
    
    torch.onnx.export(
        actor,
        dummy_input,
        onnx_path,
        input_names=['market_state'],
        output_names=['strategy_logits', 'position_mean', 'position_std', 'value'],
        dynamic_axes={
            'market_state': {0: 'batch_size'},
            'strategy_logits': {0: 'batch_size'},
            'position_mean': {0: 'batch_size'},
            'position_std': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    # Create symlinks for latest model
    latest_onnx = output_dir / "latest_ppo_agent.onnx"
    if latest_onnx.exists():
        latest_onnx.unlink()
    latest_onnx.symlink_to(onnx_path.name)
    
    print("\n" + "=" * 60)
    print("PPO TRAINING COMPLETED")
    print("=" * 60)
    print(f"PyTorch model: {torch_path}")
    print(f"ONNX model: {onnx_path}")
    print(f"Latest ONNX: {latest_onnx}")
    print(f"Final average reward: {np.mean(epoch_rewards):.3f}")

def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for trading")
    
    parser.add_argument("--output-dir", default="models/rl",
                      help="Output directory for trained models")
    parser.add_argument("--epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--episodes", type=int, default=1000,
                      help="Number of episodes to generate")
    
    args = parser.parse_args()
    
    try:
        train_ppo_agent(args)
    except Exception as e:
        print(f"\nERROR: Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
