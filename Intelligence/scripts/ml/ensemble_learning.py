#!/usr/bin/env python3
"""
Ensemble Learning for Multi-Strategy Combination
Part of the 11-script ML system restoration
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import json
import os
from typing import Dict, List

class EnsembleLearning:
    """Ensemble multiple ML models for robust predictions"""
    
    def __init__(self):
        # Initialize base models
        self.base_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'linear': LinearRegression(),
            'svr': SVR(kernel='rbf', C=1.0)
        }
        
        # Voting ensemble
        self.ensemble = VotingRegressor([
            ('rf', self.base_models['rf']),
            ('linear', self.base_models['linear']),
            ('svr', self.base_models['svr'])
        ])
        
        self.is_trained = False
        self.performance_metrics = {}
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train ensemble model"""
        
        print("[ENSEMBLE] Training ensemble models...")
        
        # Train ensemble
        self.ensemble.fit(X, y)
        
        # Train individual models for comparison
        for name, model in self.base_models.items():
            model.fit(X, y)
            
            # Calculate performance
            train_score = model.score(X, y)
            self.performance_metrics[name] = {
                'train_r2': train_score,
                'model_type': type(model).__name__
            }
        
        # Ensemble performance
        ensemble_score = self.ensemble.score(X, y)
        self.performance_metrics['ensemble'] = {
            'train_r2': ensemble_score,
            'model_type': 'VotingRegressor'
        }
        
        self.is_trained = True
        print(f"[ENSEMBLE] Training complete. Ensemble R²: {ensemble_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.ensemble.predict(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Dict:
        """Predict with uncertainty from model disagreement"""
        
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Get predictions from all base models
        predictions = {}
        for name, model in self.base_models.items():
            predictions[name] = model.predict(X)
        
        # Ensemble prediction
        ensemble_pred = self.ensemble.predict(X)
        
        # Calculate uncertainty from disagreement
        pred_array = np.array(list(predictions.values()))
        uncertainty = np.std(pred_array, axis=0)
        
        return {
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'uncertainty': uncertainty,
            'confidence': 1.0 / (1.0 + uncertainty)
        }
    
    def save(self, path: str = 'models/ml/ensemble_model.pkl'):
        """Save ensemble model"""
        
        import joblib
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'ensemble': self.ensemble,
            'base_models': self.base_models,
            'performance_metrics': self.performance_metrics,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        print(f"[ENSEMBLE] Model saved to {path}")

# Main execution
if __name__ == "__main__":
    print("[ENSEMBLE] Initializing ensemble learning system...")
    
    # Generate synthetic data
    X = np.random.randn(1000, 43)
    y = np.random.randn(1000) * 10
    
    # Add signal structure
    y += X[:, 0] * 0.5 + X[:, 5] * 0.3 + np.sin(X[:, 10]) * 0.2
    
    # Initialize and train
    ensemble = EnsembleLearning()
    ensemble.train(X, y)
    
    # Test predictions
    X_test = np.random.randn(10, 43)
    results = ensemble.predict_with_uncertainty(X_test)
    
    print(f"[ENSEMBLE] Test predictions: {results['ensemble_prediction'][:5]}")
    print(f"[ENSEMBLE] Uncertainties: {results['uncertainty'][:5]}")
    
    # Save model
    ensemble.save()
    print("[ENSEMBLE] Ensemble learning complete!")