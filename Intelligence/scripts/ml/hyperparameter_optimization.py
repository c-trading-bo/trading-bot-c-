#!/usr/bin/env python3
"""
Automated Hyperparameter Optimization for ML Models
Part of the 11-script ML system restoration
"""

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import json
import os
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    """Automated hyperparameter optimization using Optuna"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.study = None
        self.best_params = None
        self.optimization_history = []
        
        # Supported models and their parameter spaces
        self.parameter_spaces = {
            'random_forest': {
                'n_estimators': (50, 500),
                'max_depth': (3, 20),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['auto', 'sqrt', 'log2']
            }
        }
    
    def objective(self, trial, X: np.ndarray, y: np.ndarray):
        """Objective function for optimization"""
        
        if self.model_type == 'random_forest':
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
                'random_state': 42
            }
            
            # Create model
            model = RandomForestRegressor(**params)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y, 
                cv=5, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Return negative MSE (higher is better for Optuna)
            return np.mean(cv_scores)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def optimize(self, X: np.ndarray, y: np.ndarray, n_trials: int = 100) -> Dict:
        """Run hyperparameter optimization"""
        
        print(f"[HYPERPARAM] Starting optimization for {self.model_type}...")
        print(f"[HYPERPARAM] Running {n_trials} trials with {X.shape[0]} samples")
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',  # Maximize negative MSE
            study_name=f'{self.model_type}_optimization'
        )
        
        # Optimize
        objective_with_data = lambda trial: self.objective(trial, X, y)
        self.study.optimize(objective_with_data, n_trials=n_trials)
        
        # Store best parameters
        self.best_params = self.study.best_params
        
        # Store optimization history
        optimization_result = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials),
            'optimization_completed': True
        }
        
        self.optimization_history.append(optimization_result)
        
        print(f"[HYPERPARAM] Optimization complete!")
        print(f"[HYPERPARAM] Best score: {self.study.best_value:.6f}")
        print(f"[HYPERPARAM] Best params: {self.best_params}")
        
        return optimization_result
    
    def train_best_model(self, X: np.ndarray, y: np.ndarray):
        """Train model with best parameters"""
        
        if not self.best_params:
            raise ValueError("No optimization performed yet")
        
        if self.model_type == 'random_forest':
            model = RandomForestRegressor(**self.best_params)
            model.fit(X, y)
            return model
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def compare_with_default(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Compare optimized vs default parameters"""
        
        if not self.best_params:
            raise ValueError("No optimization performed yet")
        
        # Default model
        if self.model_type == 'random_forest':
            default_model = RandomForestRegressor(random_state=42)
            default_scores = cross_val_score(
                default_model, X, y, 
                cv=5, 
                scoring='neg_mean_squared_error'
            )
            
            # Optimized model
            optimized_model = RandomForestRegressor(**self.best_params)
            optimized_scores = cross_val_score(
                optimized_model, X, y, 
                cv=5, 
                scoring='neg_mean_squared_error'
            )
            
            comparison = {
                'default_score': np.mean(default_scores),
                'optimized_score': np.mean(optimized_scores),
                'improvement': np.mean(optimized_scores) - np.mean(default_scores),
                'improvement_pct': ((np.mean(optimized_scores) - np.mean(default_scores)) / 
                                   abs(np.mean(default_scores))) * 100
            }
            
            print(f"[HYPERPARAM] Default MSE: {-comparison['default_score']:.6f}")
            print(f"[HYPERPARAM] Optimized MSE: {-comparison['optimized_score']:.6f}")
            print(f"[HYPERPARAM] Improvement: {comparison['improvement_pct']:.2f}%")
            
            return comparison
    
    def get_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Get feature importance from optimized model"""
        
        if not self.best_params:
            raise ValueError("No optimization performed yet")
        
        model = self.train_best_model(X, y)
        
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
            
            # Sort by importance
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            importance_dict = dict(zip(feature_names, importance_scores))
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            return {
                'feature_importance': dict(sorted_importance),
                'top_features': [name for name, _ in sorted_importance[:10]]
            }
        
        return {'error': 'Model does not support feature importance'}
    
    def save_optimization_results(self, path: str = 'models/ml/hyperparameter_optimization.json'):
        """Save optimization results"""
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_data = {
            'optimization_history': self.optimization_history,
            'current_best_params': self.best_params,
            'model_type': self.model_type,
            'parameter_spaces': self.parameter_spaces
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"[HYPERPARAM] Results saved to {path}")

class MultiModelOptimizer:
    """Optimize multiple model types simultaneously"""
    
    def __init__(self):
        self.optimizers = {}
        self.comparison_results = {}
    
    def optimize_all_models(self, X: np.ndarray, y: np.ndarray, 
                           models: List[str] = ['random_forest'], 
                           n_trials: int = 50):
        """Optimize all specified models"""
        
        print(f"[MULTI-OPTIMIZER] Starting optimization for {len(models)} model types...")
        
        for model_type in models:
            print(f"\n[MULTI-OPTIMIZER] Optimizing {model_type}...")
            
            optimizer = HyperparameterOptimizer(model_type)
            result = optimizer.optimize(X, y, n_trials)
            comparison = optimizer.compare_with_default(X, y)
            
            self.optimizers[model_type] = optimizer
            self.comparison_results[model_type] = {
                'optimization_result': result,
                'comparison': comparison
            }
        
        # Find best model
        best_model = max(
            self.comparison_results.keys(),
            key=lambda k: self.comparison_results[k]['optimization_result']['best_score']
        )
        
        print(f"\n[MULTI-OPTIMIZER] Best model: {best_model}")
        print(f"[MULTI-OPTIMIZER] Best score: {self.comparison_results[best_model]['optimization_result']['best_score']:.6f}")
        
        return {
            'best_model': best_model,
            'all_results': self.comparison_results
        }
    
    def save_all_results(self, path: str = 'models/ml/multi_model_optimization.json'):
        """Save all optimization results"""
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.comparison_results, f, indent=2)
        
        print(f"[MULTI-OPTIMIZER] All results saved to {path}")

# Main execution
if __name__ == "__main__":
    print("[HYPERPARAM] Initializing hyperparameter optimization...")
    
    # Generate synthetic data
    X = np.random.randn(1000, 43)
    y = np.random.randn(1000) * 10
    
    # Add signal structure
    y += X[:, 0] * 0.5 + X[:, 5] * 0.3 + np.sin(X[:, 10]) * 0.2
    
    # Single model optimization
    optimizer = HyperparameterOptimizer('random_forest')
    result = optimizer.optimize(X, y, n_trials=50)
    comparison = optimizer.compare_with_default(X, y)
    
    # Feature importance
    importance = optimizer.get_feature_importance(X, y)
    print(f"[HYPERPARAM] Top 5 features: {importance['top_features'][:5]}")
    
    # Save results
    optimizer.save_optimization_results()
    
    print("[HYPERPARAM] Hyperparameter optimization complete!")