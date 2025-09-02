#!/usr/bin/env python3
"""
ML Model Performance Monitoring and Validation
Part of the 11-script ML system restoration
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceMonitor:
    """Monitor and validate ML model performance in production"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.prediction_history = []
        self.performance_metrics = []
        self.drift_alerts = []
        
        # Performance thresholds
        self.mse_threshold = 100.0
        self.mae_threshold = 10.0
        self.r2_threshold = 0.1
        
        # Drift detection parameters
        self.drift_window_size = 100
        self.drift_significance = 0.05
        
    def log_prediction(self, features: np.ndarray, prediction: float, 
                      actual: float = None, timestamp: datetime = None):
        """Log a single prediction for monitoring"""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        prediction_entry = {
            'timestamp': timestamp.isoformat(),
            'features': features.tolist() if isinstance(features, np.ndarray) else features,
            'prediction': prediction,
            'actual': actual,
            'error': actual - prediction if actual is not None else None
        }
        
        self.prediction_history.append(prediction_entry)
        
        # Keep only recent history (last 10000 predictions)
        if len(self.prediction_history) > 10000:
            self.prediction_history = self.prediction_history[-10000:]
    
    def calculate_performance_metrics(self, window_size: int = 1000) -> Dict:
        """Calculate performance metrics for recent predictions"""
        
        # Get recent predictions with actual values
        recent_predictions = [
            p for p in self.prediction_history[-window_size:] 
            if p['actual'] is not None
        ]
        
        if len(recent_predictions) < 10:
            return {'status': 'insufficient_data', 'sample_size': len(recent_predictions)}
        
        # Extract predictions and actuals
        predictions = [p['prediction'] for p in recent_predictions]
        actuals = [p['actual'] for p in recent_predictions]
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        rmse = np.sqrt(mse)
        
        # Additional metrics
        residuals = np.array(actuals) - np.array(predictions)
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Percentage errors
        mape = np.mean(np.abs(residuals / np.array(actuals))) * 100
        
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'sample_size': len(recent_predictions),
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2_score': r2,
            'mean_residual': mean_residual,
            'std_residual': std_residual,
            'mape': mape,
            'performance_status': self._assess_performance(mse, mae, r2)
        }
        
        self.performance_metrics.append(metrics)
        
        return metrics
    
    def _assess_performance(self, mse: float, mae: float, r2: float) -> str:
        """Assess overall performance status"""
        
        if (mse > self.mse_threshold or 
            mae > self.mae_threshold or 
            r2 < self.r2_threshold):
            return 'POOR'
        elif (mse < self.mse_threshold * 0.5 and 
              mae < self.mae_threshold * 0.5 and 
              r2 > self.r2_threshold * 2):
            return 'EXCELLENT'
        else:
            return 'GOOD'
    
    def detect_distribution_drift(self, feature_idx: int = 0) -> Dict:
        """Detect distribution drift in input features"""
        
        if len(self.prediction_history) < self.drift_window_size * 2:
            return {'status': 'insufficient_data'}
        
        # Get feature values from recent history
        recent_features = [
            p['features'][feature_idx] for p in self.prediction_history[-self.drift_window_size:]
            if len(p['features']) > feature_idx
        ]
        
        historical_features = [
            p['features'][feature_idx] for p in self.prediction_history[-2*self.drift_window_size:-self.drift_window_size]
            if len(p['features']) > feature_idx
        ]
        
        if len(recent_features) < 50 or len(historical_features) < 50:
            return {'status': 'insufficient_data'}
        
        # Kolmogorov-Smirnov test for distribution drift
        ks_statistic, p_value = stats.ks_2samp(historical_features, recent_features)
        
        # t-test for mean shift
        t_statistic, t_p_value = stats.ttest_ind(historical_features, recent_features)
        
        drift_detected = p_value < self.drift_significance
        
        drift_result = {
            'feature_idx': feature_idx,
            'ks_statistic': ks_statistic,
            'ks_p_value': p_value,
            't_statistic': t_statistic,
            't_p_value': t_p_value,
            'drift_detected': drift_detected,
            'historical_mean': np.mean(historical_features),
            'recent_mean': np.mean(recent_features),
            'mean_shift': np.mean(recent_features) - np.mean(historical_features)
        }
        
        if drift_detected:
            self._log_drift_alert(drift_result)
        
        return drift_result
    
    def detect_performance_drift(self) -> Dict:
        """Detect drift in model performance"""
        
        if len(self.performance_metrics) < 10:
            return {'status': 'insufficient_data'}
        
        # Get recent performance metrics
        recent_metrics = self.performance_metrics[-5:]
        historical_metrics = self.performance_metrics[-10:-5] if len(self.performance_metrics) >= 10 else []
        
        if not historical_metrics:
            return {'status': 'insufficient_historical_data'}
        
        # Compare MSE trends
        recent_mse = np.mean([m['mse'] for m in recent_metrics])
        historical_mse = np.mean([m['mse'] for m in historical_metrics])
        
        # Compare R² trends
        recent_r2 = np.mean([m['r2_score'] for m in recent_metrics])
        historical_r2 = np.mean([m['r2_score'] for m in historical_metrics])
        
        # Detect significant degradation
        mse_degradation = (recent_mse - historical_mse) / historical_mse if historical_mse > 0 else 0
        r2_degradation = (historical_r2 - recent_r2) / abs(historical_r2) if historical_r2 != 0 else 0
        
        performance_drift = mse_degradation > 0.2 or r2_degradation > 0.2
        
        performance_drift_result = {
            'performance_drift_detected': performance_drift,
            'mse_degradation_pct': mse_degradation * 100,
            'r2_degradation_pct': r2_degradation * 100,
            'recent_mse': recent_mse,
            'historical_mse': historical_mse,
            'recent_r2': recent_r2,
            'historical_r2': historical_r2
        }
        
        if performance_drift:
            self._log_drift_alert(performance_drift_result)
        
        return performance_drift_result
    
    def _log_drift_alert(self, drift_info: Dict):
        """Log drift alert"""
        
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'model_name': self.model_name,
            'alert_type': 'DRIFT_DETECTED',
            'drift_info': drift_info
        }
        
        self.drift_alerts.append(alert)
        print(f"[DRIFT ALERT] {self.model_name}: Drift detected - {drift_info}")
    
    def generate_monitoring_report(self) -> Dict:
        """Generate comprehensive monitoring report"""
        
        # Recent performance
        recent_performance = self.calculate_performance_metrics()
        
        # Distribution drift for key features
        feature_drift_results = []
        for i in range(min(5, 43)):  # Check first 5 features
            drift_result = self.detect_distribution_drift(i)
            if drift_result.get('status') != 'insufficient_data':
                feature_drift_results.append(drift_result)
        
        # Performance drift
        performance_drift = self.detect_performance_drift()
        
        # Summary statistics
        total_predictions = len(self.prediction_history)
        predictions_with_actuals = len([p for p in self.prediction_history if p['actual'] is not None])
        recent_alerts = len([a for a in self.drift_alerts 
                           if datetime.fromisoformat(a['timestamp']) > datetime.utcnow() - timedelta(days=7)])
        
        report = {
            'model_name': self.model_name,
            'report_timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_predictions': total_predictions,
                'predictions_with_actuals': predictions_with_actuals,
                'recent_alerts': recent_alerts,
                'overall_status': recent_performance.get('performance_status', 'UNKNOWN')
            },
            'recent_performance': recent_performance,
            'feature_drift': feature_drift_results,
            'performance_drift': performance_drift,
            'recent_drift_alerts': self.drift_alerts[-5:] if self.drift_alerts else []
        }
        
        return report
    
    def save_monitoring_data(self, path: str = None):
        """Save monitoring data to file"""
        
        if path is None:
            path = f'models/ml/monitoring_{self.model_name}.json'
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        monitoring_data = {
            'model_name': self.model_name,
            'prediction_history': self.prediction_history[-1000:],  # Keep last 1000
            'performance_metrics': self.performance_metrics,
            'drift_alerts': self.drift_alerts,
            'monitoring_report': self.generate_monitoring_report()
        }
        
        with open(path, 'w') as f:
            json.dump(monitoring_data, f, indent=2, default=str)
        
        print(f"[MONITOR] Monitoring data saved to {path}")

# Main execution
if __name__ == "__main__":
    print("[MONITOR] Initializing ML model performance monitoring...")
    
    # Initialize monitor
    monitor = ModelPerformanceMonitor('test_model')
    
    # Simulate model predictions over time
    for i in range(500):
        # Generate synthetic features and predictions
        features = np.random.randn(43)
        
        # Simulate model prediction with some error
        true_value = np.sum(features[:5]) + np.random.randn() * 0.5
        prediction = true_value + np.random.randn() * 2.0  # Add prediction error
        
        # Add concept drift after prediction 300
        if i > 300:
            features = features + 0.5  # Shift feature distribution
            prediction = prediction + 1.0  # Shift predictions
        
        # Log prediction
        monitor.log_prediction(features, prediction, true_value)
        
        # Calculate metrics every 50 predictions
        if (i + 1) % 50 == 0:
            metrics = monitor.calculate_performance_metrics()
            if metrics.get('status') != 'insufficient_data':
                print(f"Prediction {i+1}: MSE={metrics['mse']:.4f}, "
                      f"R²={metrics['r2_score']:.4f}, "
                      f"Status={metrics['performance_status']}")
    
    # Generate final report
    report = monitor.generate_monitoring_report()
    print(f"\n[MONITOR] Final Report:")
    print(f"  Total Predictions: {report['summary']['total_predictions']}")
    print(f"  Overall Status: {report['summary']['overall_status']}")
    print(f"  Recent Alerts: {report['summary']['recent_alerts']}")
    
    # Save monitoring data
    monitor.save_monitoring_data()
    print("[MONITOR] Model performance monitoring complete!")