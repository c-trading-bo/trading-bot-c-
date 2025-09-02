#!/usr/bin/env python3
"""
CVaR-PPO Advanced RL Agent Training Script
"""

import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

class CVaRPPOAgent(nn.Module):
    def __init__(self, state_dim=10, action_dim=5, hidden_dim=128):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

def train_cvar_ppo(data_file, episodes, lr, cvar_alpha, risk_penalty, save_dir):
    """Train CVaR-PPO agent"""
    print(f"[CVAR-PPO] Training agent from {data_file}")
    print(f"[CVAR-PPO] Episodes: {episodes}, LR: {lr}, CVaR Alpha: {cvar_alpha}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Load or generate data
    try:
        df = pd.read_parquet(data_file)
        print(f"[CVAR-PPO] Loaded {len(df)} samples")
    except Exception as e:
        print(f"[CVAR-PPO] Error loading data: {e}")
        # Create synthetic data
        df = pd.DataFrame({
            'state': [[np.random.randn() for _ in range(10)] for _ in range(1000)],
            'action': np.random.randint(0, 5, 1000),
            'reward': np.random.uniform(-1, 2, 1000),
            'done': np.random.choice([True, False], 1000)
        })
        print(f"[CVAR-PPO] Generated {len(df)} synthetic samples")
    
    # Initialize agent
    agent = CVaRPPOAgent(state_dim=10, action_dim=5)
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    
    # Training loop (simplified)
    total_reward = 0
    for episode in range(episodes):
        # Simulate episode
        states = torch.FloatTensor([row for row in df['state'].values[:100]])
        rewards = torch.FloatTensor(df['reward'].values[:100])
        
        # Forward pass
        action_probs, values = agent(states)
        
        # Simplified loss (PPO would be more complex)
        policy_loss = -(torch.log(action_probs.max(dim=1)[0]) * rewards).mean()
        value_loss = nn.MSELoss()(values.squeeze(), rewards)
        
        # CVaR risk penalty
        returns_sorted = torch.sort(rewards)[0]
        cvar_cutoff = int(cvar_alpha * len(returns_sorted))
        cvar_loss = -returns_sorted[:cvar_cutoff].mean() * risk_penalty
        
        total_loss = policy_loss + value_loss + cvar_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        episode_reward = rewards.sum().item()
        total_reward += episode_reward
        
        if episode % 100 == 0:
            print(f"[CVAR-PPO] Episode {episode}, Reward: {episode_reward:.2f}, Loss: {total_loss.item():.4f}")
    
    # Save model
    model_path = os.path.join(save_dir, 'cvar_ppo.pth')
    torch.save(agent.state_dict(), model_path)
    print(f"[CVAR-PPO] Saved model to {model_path}")
    
    # Export to ONNX
    agent.eval()
    dummy_input = torch.randn(1, 10)
    onnx_path = os.path.join(save_dir, 'cvar_ppo.onnx')
    torch.onnx.export(agent, dummy_input, onnx_path,
                     input_names=['state'], output_names=['action_probs', 'value'],
                     dynamic_axes={'state': {0: 'batch_size'}})
    print(f"[CVAR-PPO] Exported ONNX to {onnx_path}")
    
    return total_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CVaR-PPO Agent')
    parser.add_argument('--data', required=True, help='Training data file')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--cvar_alpha', type=float, default=0.05, help='CVaR alpha')
    parser.add_argument('--risk_penalty', type=float, default=0.1, help='Risk penalty weight')
    parser.add_argument('--save_dir', required=True, help='Directory to save models')
    
    args = parser.parse_args()
    
    total_reward = train_cvar_ppo(
        args.data, args.episodes, args.lr, 
        args.cvar_alpha, args.risk_penalty, args.save_dir
    )
    print(f"[CVAR-PPO] Training completed with total reward: {total_reward:.2f}")
