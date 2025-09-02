#!/usr/bin/env python3
"""
Online Learning System for Real-Time Adaptation
Part of the 11-script ML system restoration
"""

import numpy as np
from sklearn.linear_model import SGDRegressor
from collections import deque
import json
import os
from typing import Dict, Tuple

class OnlineLearningSystem:
    """Real-time adaptive learning system"""
    
    def __init__(self, buffer_size: int = 1000):
        # Online models
        self.sgd_model = SGDRegressor(
            learning_rate='adaptive',
            eta0=0.01,
            alpha=0.001,
            random_state=42
        )
        
        # Sliding window buffer
        self.buffer = deque(maxlen=buffer_size)
        self.is_initialized = False
        
        # Performance tracking
        self.performance_history = []
        self.adaptation_count = 0
        
        # Drift detection
        self.drift_threshold = 2.0
        self.recent_errors = deque(maxlen=50)
        
    def partial_fit(self, X: np.ndarray, y: np.ndarray):
        """Incrementally update model with new data"""
        
        # Store in buffer
        for features, target in zip(X, y):
            self.buffer.append({'features': features, 'target': target})
        
        # Initialize model if needed
        if not self.is_initialized:
            self.sgd_model.fit(X, y)
            self.is_initialized = True
            print("[ONLINE] Model initialized")
        else:
            # Incremental learning
            self.sgd_model.partial_fit(X, y)
            self.adaptation_count += 1
            
            # Check for concept drift
            if len(self.recent_errors) >= 10:
                self._check_concept_drift()
        
        # Calculate recent performance
        if len(self.buffer) >= 10:
            self._update_performance_metrics()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        
        if not self.is_initialized:
            # Return zeros if not initialized
            return np.zeros(X.shape[0])
        
        return self.sgd_model.predict(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence based on recent performance"""
        
        predictions = self.predict(X)
        
        # Calculate confidence based on recent error variance
        if len(self.recent_errors) > 5:
            error_std = np.std(self.recent_errors)
            confidence = 1.0 / (1.0 + error_std)
        else:
            confidence = 0.5  # Default moderate confidence
        
        confidence_array = np.full(predictions.shape, confidence)
        
        return predictions, confidence_array
    
    def _check_concept_drift(self):
        """Detect concept drift in the data stream"""
        
        recent_error = np.mean(self.recent_errors[-10:])
        historical_error = np.mean(self.recent_errors[:-10]) if len(self.recent_errors) > 10 else recent_error
        
        # Drift detected if recent error significantly higher
        if recent_error > historical_error * self.drift_threshold:
            print(f"[ONLINE] Concept drift detected! Recent error: {recent_error:.4f}")
            self._adapt_to_drift()
    
    def _adapt_to_drift(self):
        """Adapt model to concept drift"""
        
        # Reset learning rate for faster adaptation
        self.sgd_model.set_params(eta0=0.05)
        
        # Retrain on recent buffer data if enough samples
        if len(self.buffer) >= 100:
            recent_data = list(self.buffer)[-100:]
            X_recent = np.array([d['features'] for d in recent_data])
            y_recent = np.array([d['target'] for d in recent_data])
            
            # Partial refit on recent data
            self.sgd_model.partial_fit(X_recent, y_recent)
            
            print("[ONLINE] Adapted to concept drift")
        
        # Reset learning rate
        self.sgd_model.set_params(eta0=0.01)
    
    def _update_performance_metrics(self):
        """Update performance tracking"""
        
        if len(self.buffer) < 20:
            return
        
        # Get recent data for evaluation
        recent_data = list(self.buffer)[-20:]
        X_recent = np.array([d['features'] for d in recent_data])
        y_recent = np.array([d['target'] for d in recent_data])
        
        # Calculate prediction error
        if self.is_initialized:
            y_pred = self.predict(X_recent)
            mse = np.mean((y_recent - y_pred) ** 2)
            mae = np.mean(np.abs(y_recent - y_pred))
            
            # Store error for drift detection
            self.recent_errors.append(mse)
            
            # Store performance metrics
            self.performance_history.append({
                'adaptation_count': self.adaptation_count,
                'mse': mse,
                'mae': mae,
                'buffer_size': len(self.buffer)
            })
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        
        if not self.performance_history:
            return {'status': 'no_data'}
        
        recent_metrics = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        return {
            'adaptations': self.adaptation_count,
            'recent_mse': np.mean([m['mse'] for m in recent_metrics]),
            'recent_mae': np.mean([m['mae'] for m in recent_metrics]),
            'buffer_utilization': len(self.buffer) / self.buffer.maxlen,
            'drift_alerts': sum(1 for e in self.recent_errors if e > np.mean(self.recent_errors) * self.drift_threshold)
        }
    
    def save(self, path: str = 'models/ml/online_learning.pkl'):
        """Save online learning model"""
        
        import joblib
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'sgd_model': self.sgd_model,
            'performance_history': self.performance_history,
            'adaptation_count': self.adaptation_count,
            'is_initialized': self.is_initialized
        }
        
        joblib.dump(model_data, path)
        print(f"[ONLINE] Model saved to {path}")

# Main execution
if __name__ == "__main__":
    print("[ONLINE] Initializing online learning system...")
    
    # Initialize system
    online_learner = OnlineLearningSystem()
    
    # Simulate streaming data
    for batch in range(10):
        # Generate batch of data
        X_batch = np.random.randn(20, 43)
        y_batch = np.random.randn(20) * 10
        
        # Add concept drift after batch 5
        if batch > 5:
            y_batch += 5  # Shift target distribution
        
        # Online learning
        online_learner.partial_fit(X_batch, y_batch)
        
        # Test prediction
        X_test = np.random.randn(5, 43)
        predictions, confidence = online_learner.predict_with_confidence(X_test)
        
        if batch % 3 == 0:
            summary = online_learner.get_performance_summary()
            print(f"Batch {batch}: Adaptations={summary.get('adaptations', 0)}, "
                  f"MSE={summary.get('recent_mse', 0):.4f}")
    
    # Save model
    online_learner.save()
    print("[ONLINE] Online learning system complete!")