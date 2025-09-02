#!/usr/bin/env python3
"""
Advanced Portfolio Optimization using ML
Part of the 11-script ML system restoration
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import json
import os
from typing import Dict, List, Tuple

class MLPortfolioOptimizer:
    """Machine Learning enhanced portfolio optimization"""
    
    def __init__(self, risk_aversion: float = 1.0):
        self.risk_aversion = risk_aversion
        self.returns_history = []
        self.weights_history = []
        self.volatility_models = {}
        
        # Risk models
        self.covariance_estimator = LedoitWolf()
        
        # Constraints
        self.min_weight = 0.0
        self.max_weight = 0.5
        self.max_concentration = 0.3
        
    def add_returns_data(self, returns: np.ndarray, symbols: List[str]):
        """Add returns data for optimization"""
        
        self.returns_history.append({
            'returns': returns,
            'symbols': symbols,
            'timestamp': pd.Timestamp.now()
        })
        
        # Keep only recent history
        if len(self.returns_history) > 100:
            self.returns_history = self.returns_history[-100:]
    
    def estimate_expected_returns(self, symbols: List[str], lookback: int = 30) -> np.ndarray:
        """Estimate expected returns using ML techniques"""
        
        if not self.returns_history:
            return np.zeros(len(symbols))
        
        # Get recent returns
        recent_returns = []
        for entry in self.returns_history[-lookback:]:
            if entry['symbols'] == symbols:
                recent_returns.append(entry['returns'])
        
        if not recent_returns:
            return np.zeros(len(symbols))
        
        returns_matrix = np.array(recent_returns)
        
        # Simple momentum + mean reversion combination
        momentum_returns = np.mean(returns_matrix[-5:], axis=0) if len(returns_matrix) >= 5 else np.zeros(len(symbols))
        mean_reversion = -np.mean(returns_matrix, axis=0) * 0.1
        
        # Combine with decay
        expected_returns = 0.7 * momentum_returns + 0.3 * mean_reversion
        
        return expected_returns
    
    def estimate_covariance_matrix(self, symbols: List[str], lookback: int = 50) -> np.ndarray:
        """Estimate covariance matrix with ML shrinkage"""
        
        if not self.returns_history:
            return np.eye(len(symbols)) * 0.01  # Default small variance
        
        # Get recent returns
        recent_returns = []
        for entry in self.returns_history[-lookback:]:
            if entry['symbols'] == symbols:
                recent_returns.append(entry['returns'])
        
        if len(recent_returns) < 10:
            return np.eye(len(symbols)) * 0.01
        
        returns_matrix = np.array(recent_returns)
        
        # Use Ledoit-Wolf shrinkage
        cov_matrix, _ = self.covariance_estimator.fit(returns_matrix).covariance_, self.covariance_estimator.shrinkage_
        
        return cov_matrix
    
    def optimize_portfolio(self, symbols: List[str], target_return: float = None) -> Dict:
        """Optimize portfolio weights"""
        
        n_assets = len(symbols)
        
        # Estimate inputs
        expected_returns = self.estimate_expected_returns(symbols)
        cov_matrix = self.estimate_covariance_matrix(symbols)
        
        # Objective function (maximize utility = return - 0.5 * risk_aversion * variance)
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance
            return -utility  # Minimize negative utility
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({
                'type': 'eq', 
                'fun': lambda w: np.dot(w, expected_returns) - target_return
            })
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_guess = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
        else:
            print("[PORTFOLIO] Optimization failed, using equal weights")
            optimal_weights = initial_guess
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
        
        # Concentration risk
        concentration = np.max(optimal_weights)
        
        optimization_result = {
            'weights': optimal_weights.tolist(),
            'symbols': symbols,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'concentration': concentration,
            'optimization_success': result.success
        }
        
        # Store in history
        self.weights_history.append(optimization_result)
        
        return optimization_result
    
    def calculate_risk_attribution(self, weights: np.ndarray, symbols: List[str]) -> Dict:
        """Calculate risk attribution by asset"""
        
        cov_matrix = self.estimate_covariance_matrix(symbols)
        
        # Marginal risk contribution
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        marginal_contrib = np.dot(cov_matrix, weights)
        
        # Component contributions
        risk_contrib = weights * marginal_contrib
        risk_contrib_pct = risk_contrib / np.sum(risk_contrib) if np.sum(risk_contrib) > 0 else np.zeros_like(risk_contrib)
        
        return {
            'symbols': symbols,
            'weights': weights.tolist(),
            'risk_contributions': risk_contrib.tolist(),
            'risk_contributions_pct': risk_contrib_pct.tolist(),
            'total_portfolio_risk': np.sqrt(portfolio_variance)
        }
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization history"""
        
        if not self.weights_history:
            return {'status': 'no_optimizations'}
        
        recent_opts = self.weights_history[-10:]
        
        avg_return = np.mean([opt['expected_return'] for opt in recent_opts])
        avg_volatility = np.mean([opt['volatility'] for opt in recent_opts])
        avg_sharpe = np.mean([opt['sharpe_ratio'] for opt in recent_opts])
        success_rate = np.mean([opt['optimization_success'] for opt in recent_opts])
        
        return {
            'total_optimizations': len(self.weights_history),
            'recent_avg_return': avg_return,
            'recent_avg_volatility': avg_volatility,
            'recent_avg_sharpe': avg_sharpe,
            'optimization_success_rate': success_rate
        }
    
    def save(self, path: str = 'models/ml/portfolio_optimizer.json'):
        """Save portfolio optimization results"""
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        save_data = {
            'weights_history': self.weights_history,
            'optimization_summary': self.get_optimization_summary(),
            'risk_aversion': self.risk_aversion,
            'constraints': {
                'min_weight': self.min_weight,
                'max_weight': self.max_weight,
                'max_concentration': self.max_concentration
            }
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"[PORTFOLIO] Optimization results saved to {path}")

# Main execution
if __name__ == "__main__":
    print("[PORTFOLIO] Initializing ML portfolio optimizer...")
    
    # Initialize optimizer
    optimizer = MLPortfolioOptimizer(risk_aversion=2.0)
    
    # Simulate portfolio optimization
    symbols = ['ES', 'NQ', 'YM', 'RTY']
    
    # Add synthetic returns data
    for i in range(50):
        returns = np.random.randn(len(symbols)) * 0.02  # 2% daily volatility
        optimizer.add_returns_data(returns, symbols)
    
    # Optimize portfolio
    result = optimizer.optimize_portfolio(symbols)
    
    print(f"[PORTFOLIO] Optimal weights: {dict(zip(symbols, result['weights']))}")
    print(f"[PORTFOLIO] Expected return: {result['expected_return']:.4f}")
    print(f"[PORTFOLIO] Volatility: {result['volatility']:.4f}")
    print(f"[PORTFOLIO] Sharpe ratio: {result['sharpe_ratio']:.4f}")
    
    # Risk attribution
    risk_attr = optimizer.calculate_risk_attribution(np.array(result['weights']), symbols)
    print(f"[PORTFOLIO] Risk contributions: {dict(zip(symbols, risk_attr['risk_contributions_pct']))}")
    
    # Save results
    optimizer.save()
    print("[PORTFOLIO] Portfolio optimization complete!")