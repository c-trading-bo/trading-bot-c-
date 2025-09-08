#!/usr/bin/env python3
"""
Bayesian Uncertainty Quantification for Risk Management
Part of the 11-script ML system restoration
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Dict
import json
import os

class BayesianNN(nn.Module):
    """Bayesian Neural Network for uncertainty quantification"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super(BayesianNN, self).__init__()
        
        # Mean networks
        self.fc1_mean = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3_mean = nn.Linear(hidden_dim // 2, output_dim)
        
        # Variance networks (for uncertainty)
        self.fc1_var = nn.Linear(input_dim, hidden_dim)
        self.fc2_var = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3_var = nn.Linear(hidden_dim // 2, output_dim)
        
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        # Mean path
        mean = self.relu(self.fc1_mean(x))
        mean = self.relu(self.fc2_mean(mean))
        mean = self.fc3_mean(mean)
        
        # Variance path
        var = self.relu(self.fc1_var(x))
        var = self.relu(self.fc2_var(var))
        var = self.softplus(self.fc3_var(var)) + 1e-6  # Ensure positive
        
        return mean, var

class UncertaintyQuantifier:
    """Quantify prediction uncertainty for risk management"""
    
    def __init__(self, input_dim: int = 43):
        self.model = BayesianNN(input_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.training_data = []
        
    def predict_with_uncertainty(self, features: np.ndarray, n_samples: int = 100) -> Tuple[float, float, float]:
        """
        Predict with uncertainty bounds
        Returns: (mean_prediction, lower_bound, upper_bound)
        """
        
        x = torch.FloatTensor(features).unsqueeze(0)
        
        self.model.eval()
        with torch.no_grad():
            mean, var = self.model(x)
            
            # Sample from distribution
            dist = Normal(mean, torch.sqrt(var))
            samples = [dist.sample() for _ in range(n_samples)]
            samples = torch.cat(samples)
            
            # Calculate statistics
            mean_pred = samples.mean().item()
            std_pred = samples.std().item()
            
            # 95% confidence interval
            lower_bound = mean_pred - 1.96 * std_pred
            upper_bound = mean_pred + 1.96 * std_pred
            
        return mean_pred, lower_bound, upper_bound
    
    def calculate_risk_metrics(self, features: np.ndarray) -> Dict:
        """Calculate comprehensive risk metrics"""
        
        mean_pred, lower, upper = self.predict_with_uncertainty(features)
        
        # Calculate various risk metrics
        metrics = {
            'prediction': mean_pred,
            'confidence_interval': (lower, upper),
            'uncertainty': upper - lower,
            'relative_uncertainty': (upper - lower) / (abs(mean_pred) + 1e-6),
            'risk_score': self._calculate_risk_score(mean_pred, upper - lower),
            'suggested_position_size': self._suggest_position_size(upper - lower)
        }
        
        return metrics
    
    def _calculate_risk_score(self, prediction: float, uncertainty: float) -> float:
        """Calculate risk score from 0 (low risk) to 1 (high risk)"""
        
        # Normalize uncertainty
        normalized_uncertainty = min(uncertainty / 10, 1.0)
        
        # Consider prediction magnitude
        prediction_risk = min(abs(prediction) / 100, 1.0)
        
        # Combined risk score
        risk_score = 0.7 * normalized_uncertainty + 0.3 * prediction_risk
        
        return min(max(risk_score, 0), 1)
    
    def _suggest_position_size(self, uncertainty: float) -> float:
        """Suggest position size based on uncertainty"""
        
        if uncertainty < 2:
            return 1.0  # Full size for high confidence
        elif uncertainty < 5:
            return 0.7  # Reduced size for moderate uncertainty
        elif uncertainty < 10:
            return 0.4  # Small size for high uncertainty
        else:
            return 0.2  # Minimum size for extreme uncertainty
    
    def train(self, features: np.ndarray, target: float):
        """Train the Bayesian model"""
        
        # Store data
        self.training_data.append({'features': features, 'target': target})
        
        # Batch training
        if len(self.training_data) >= 32:
            self._train_batch()
    
    def _train_batch(self):
        """Train on batch of data"""
        
        batch = self.training_data[-32:]
        
        X = torch.FloatTensor([d['features'] for d in batch])
        y = torch.FloatTensor([d['target'] for d in batch]).unsqueeze(1)
        
        self.model.train()
        
        # Forward pass
        mean, var = self.model(X)
        
        # Negative log likelihood loss
        dist = Normal(mean, torch.sqrt(var))
        loss = -dist.log_prob(y).mean()
        
        # Add regularization for variance
        loss += 0.01 * var.mean()  # Prevent variance explosion
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, path: str = 'models/ml/uncertainty_model.pth'):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_samples': len(self.training_data)
        }, path)
        print(f"[UNCERTAINTY] Model saved to {path}")

# Main execution
if __name__ == "__main__":
    print("[UNCERTAINTY] Initializing Bayesian uncertainty quantification...")
    
    # Initialize
    uq = UncertaintyQuantifier()
    
    # Generate synthetic training data
    for i in range(200):
        features = np.random.randn(43)
        target = np.random.randn() * 10
        uq.train(features, target)
    
    # Test predictions
    test_features = np.random.randn(43)
    mean, lower, upper = uq.predict_with_uncertainty(test_features)
    metrics = uq.calculate_risk_metrics(test_features)
    
    print(f"Prediction: {mean:.2f} [{lower:.2f}, {upper:.2f}]")
    print(f"Risk Score: {metrics['risk_score']:.2f}")
    print(f"Suggested Position Size: {metrics['suggested_position_size']:.2f}")
    
    # Save model
    uq.save()
    print("[UNCERTAINTY] Model complete!")