#!/usr/bin/env python3
"""
Meta-Learning for Fast Regime Adaptation
Part of the 11-script ML system restoration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import json
from typing import Dict, List
import os

class MAML(nn.Module):
    """Model-Agnostic Meta-Learning for fast regime adaptation"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(MAML, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class MetaLearningSystem:
    """Fast adaptation to new market regimes"""
    
    def __init__(self, input_dim: int = 43):
        self.model = MAML(input_dim)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.regime_memory = {
            'bull': deque(maxlen=1000),
            'bear': deque(maxlen=1000),
            'volatile': deque(maxlen=1000),
            'ranging': deque(maxlen=1000)
        }
        self.current_regime = 'unknown'
        self.adaptation_steps = 5
        
    def detect_regime(self, features: np.ndarray) -> str:
        """Detect current market regime from features"""
        
        # Simple regime detection based on key features
        # In production, use more sophisticated methods
        volatility = features[16] if len(features) > 16 else 0
        trend = features[5] if len(features) > 5 else 0
        
        if volatility > 0.02:
            return 'volatile'
        elif trend > 0.01:
            return 'bull'
        elif trend < -0.01:
            return 'bear'
        else:
            return 'ranging'
    
    def adapt_to_regime(self, regime: str, support_data: List[Dict]):
        """Fast adaptation using meta-learning"""
        
        if not support_data:
            return
        
        # Clone model for adaptation
        adapted_model = MAML(self.model.fc1.in_features)
        adapted_model.load_state_dict(self.model.state_dict())
        
        # Fast adaptation with few examples
        adaptation_optimizer = optim.SGD(adapted_model.parameters(), lr=0.01)
        
        for _ in range(self.adaptation_steps):
            # Sample from support data
            batch = np.random.choice(support_data, min(5, len(support_data)), replace=False)
            
            X = torch.FloatTensor([d['features'] for d in batch])
            y = torch.FloatTensor([d['target'] for d in batch]).unsqueeze(1)
            
            # Adaptation step
            pred = adapted_model(X)
            loss = nn.MSELoss()(pred, y)
            
            adaptation_optimizer.zero_grad()
            loss.backward()
            adaptation_optimizer.step()
        
        # Update main model with adapted weights
        self.model.load_state_dict(adapted_model.state_dict())
        
        print(f"[META] Adapted to {regime} regime using {len(support_data)} examples")
    
    def predict_adaptive(self, features: np.ndarray) -> float:
        """Predict with regime-adaptive model"""
        
        # Detect regime
        regime = self.detect_regime(features)
        
        # Adapt if regime changed
        if regime != self.current_regime:
            support_data = list(self.regime_memory.get(regime, []))
            if support_data:
                self.adapt_to_regime(regime, support_data)
            self.current_regime = regime
        
        # Make prediction
        x = torch.FloatTensor(features).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x).item()
        
        return prediction
    
    def update_memory(self, features: np.ndarray, target: float, regime: str = None):
        """Store experience in regime-specific memory"""
        
        if regime is None:
            regime = self.detect_regime(features)
        
        self.regime_memory[regime].append({
            'features': features,
            'target': target
        })
    
    def meta_train(self, tasks: List[Dict]):
        """Meta-training on multiple tasks/regimes"""
        
        meta_loss = 0
        
        for task in tasks:
            # Clone model
            fast_model = MAML(self.model.fc1.in_features)
            fast_model.load_state_dict(self.model.state_dict())
            fast_optimizer = optim.SGD(fast_model.parameters(), lr=0.01)
            
            # Split task data
            support = task['support']
            query = task['query']
            
            # Fast adaptation on support set
            for _ in range(self.adaptation_steps):
                X_sup = torch.FloatTensor([d['features'] for d in support])
                y_sup = torch.FloatTensor([d['target'] for d in support]).unsqueeze(1)
                
                pred_sup = fast_model(X_sup)
                loss_sup = nn.MSELoss()(pred_sup, y_sup)
                
                fast_optimizer.zero_grad()
                loss_sup.backward()
                fast_optimizer.step()
            
            # Evaluate on query set
            X_qry = torch.FloatTensor([d['features'] for d in query])
            y_qry = torch.FloatTensor([d['target'] for d in query]).unsqueeze(1)
            
            pred_qry = fast_model(X_qry)
            loss_qry = nn.MSELoss()(pred_qry, y_qry)
            
            meta_loss += loss_qry
        
        # Meta-optimization
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
    
    def save(self, path: str = 'models/ml/meta_learning.pth'):
        """Save meta-learned model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'meta_optimizer_state': self.meta_optimizer.state_dict(),
            'regime_memory_sizes': {k: len(v) for k, v in self.regime_memory.items()},
            'current_regime': self.current_regime
        }, path)
        print(f"[META] Model saved to {path}")

# Main execution
if __name__ == "__main__":
    print("[META-LEARNING] Initializing fast adaptation system...")
    
    # Initialize
    meta_system = MetaLearningSystem()
    
    # Generate synthetic regime data
    regimes = ['bull', 'bear', 'volatile', 'ranging']
    
    for regime in regimes:
        for _ in range(50):
            features = np.random.randn(43)
            
            # Simulate regime-specific patterns
            if regime == 'bull':
                features[5] = abs(np.random.randn()) * 0.02  # Positive trend
            elif regime == 'bear':
                features[5] = -abs(np.random.randn()) * 0.02  # Negative trend
            elif regime == 'volatile':
                features[16] = abs(np.random.randn()) * 0.05  # High volatility
            
            target = np.random.randn() * 10
            meta_system.update_memory(features, target, regime)
    
    # Test adaptive prediction
    test_features = np.random.randn(43)
    prediction = meta_system.predict_adaptive(test_features)
    print(f"Adaptive prediction: {prediction:.2f}")
    
    # Save model
    meta_system.save()
    print("[META-LEARNING] System ready for fast regime adaptation!")