#!/usr/bin/env python3
"""
Neural Bandits Implementation for Strategy Selection
Part of the 11-script ML system restoration
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
import json
from datetime import datetime
import os

class NeuralBandit(nn.Module):
    """Neural network for contextual bandit optimization"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, n_arms: int = 12):
        super(NeuralBandit, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, n_arms)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuralBanditsOptimizer:
    """Multi-armed bandit for strategy selection with neural networks"""
    
    def __init__(self, n_features: int = 43, n_strategies: int = 12):
        self.n_features = n_features
        self.n_strategies = n_strategies
        self.model = NeuralBandit(n_features, n_arms=n_strategies)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []
        self.epsilon = 0.1  # Exploration rate
        
    def select_strategy(self, features: np.ndarray, explore: bool = True) -> Tuple[int, float]:
        """Select best strategy using Thompson sampling"""
        
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0)
        
        # Forward pass with dropout for uncertainty
        self.model.eval()
        with torch.no_grad():
            # Multiple forward passes for uncertainty estimation
            predictions = []
            for _ in range(10):
                self.model.train()  # Enable dropout
                pred = self.model(x)
                predictions.append(pred)
            
            # Calculate mean and std
            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
            
            # Thompson sampling
            if explore and np.random.random() < self.epsilon:
                # Exploration: sample from distribution
                sampled_values = mean_pred + std_pred * torch.randn_like(mean_pred)
                strategy_idx = torch.argmax(sampled_values).item()
            else:
                # Exploitation: choose best mean
                strategy_idx = torch.argmax(mean_pred).item()
            
            confidence = torch.softmax(mean_pred, dim=1)[0, strategy_idx].item()
        
        return strategy_idx, confidence
    
    def update(self, features: np.ndarray, strategy: int, reward: float):
        """Update model based on observed reward"""
        
        # Store experience
        self.memory.append({
            'features': features,
            'strategy': strategy,
            'reward': reward,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Update model if enough data
        if len(self.memory) >= 32:
            self._train_batch()
    
    def _train_batch(self):
        """Train on batch of experiences"""
        
        # Sample batch
        batch_size = min(32, len(self.memory))
        batch = np.random.choice(self.memory, batch_size, replace=False)
        
        # Prepare data
        features = torch.FloatTensor([e['features'] for e in batch])
        strategies = torch.LongTensor([e['strategy'] for e in batch])
        rewards = torch.FloatTensor([e['reward'] for e in batch])
        
        # Forward pass
        self.model.train()
        predictions = self.model(features)
        
        # Calculate loss (negative log likelihood weighted by rewards)
        log_probs = nn.functional.log_softmax(predictions, dim=1)
        selected_log_probs = log_probs.gather(1, strategies.unsqueeze(1)).squeeze()
        loss = -(selected_log_probs * rewards).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay exploration
        self.epsilon = max(0.01, self.epsilon * 0.995)
    
    def get_strategy_stats(self) -> Dict:
        """Get performance statistics for each strategy"""
        
        stats = {}
        for i in range(self.n_strategies):
            strategy_rewards = [e['reward'] for e in self.memory if e['strategy'] == i]
            if strategy_rewards:
                stats[f'S{i+1}'] = {
                    'count': len(strategy_rewards),
                    'avg_reward': np.mean(strategy_rewards),
                    'total_reward': np.sum(strategy_rewards),
                    'win_rate': np.mean([r > 0 for r in strategy_rewards])
                }
        
        return stats
    
    def save(self, path: str = 'models/ml/neural_bandits.pth'):
        """Save model and statistics"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': self.memory[-1000:],  # Keep last 1000
            'stats': self.get_strategy_stats()
        }, path)
        print(f"[BANDITS] Model saved to {path}")

# Main execution
if __name__ == "__main__":
    print("[NEURAL BANDITS] Initializing multi-armed bandit optimizer...")
    
    # Initialize
    bandit = NeuralBanditsOptimizer()
    
    # Generate synthetic data for testing
    for episode in range(100):
        # Random features
        features = np.random.randn(43)
        
        # Select strategy
        strategy, confidence = bandit.select_strategy(features)
        
        # Simulate reward (random for now)
        reward = np.random.randn() * 10
        
        # Update model
        bandit.update(features, strategy, reward)
        
        if (episode + 1) % 20 == 0:
            stats = bandit.get_strategy_stats()
            print(f"Episode {episode + 1}: {len(stats)} strategies tested")
    
    # Save model
    bandit.save()
    print("[NEURAL BANDITS] Training complete!")