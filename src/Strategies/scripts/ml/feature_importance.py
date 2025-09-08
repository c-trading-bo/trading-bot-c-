#!/usr/bin/env python3
"""
Feature Importance Analyzer for ML Feature Selection
Part of the 11-script ML system restoration
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import json
from typing import Dict, List
from datetime import datetime
import os

class FeatureImportanceAnalyzer:
    """Advanced feature selection and importance analysis"""
    
    def __init__(self, feature_names: List[str] = None):
        self.feature_names = feature_names or [f'feature_{i}' for i in range(43)]
        self.rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.importance_scores = {}
        
    def analyze_importance(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Comprehensive feature importance analysis"""
        
        print("[FEATURES] Analyzing feature importance...")
        
        # 1. Random Forest Feature Importance
        self.rf_model.fit(X, y)
        rf_importance = self.rf_model.feature_importances_
        
        # 2. Permutation Importance
        perm_importance = permutation_importance(
            self.rf_model, X, y, n_repeats=10, random_state=42
        )
        
        # 3. Correlation with target
        correlations = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
        
        # 4. Variance-based importance
        variances = np.var(X, axis=0)
        normalized_variances = variances / np.max(variances)
        
        # Combine all importance measures
        self.importance_scores = {}
        for i, name in enumerate(self.feature_names):
            self.importance_scores[name] = {
                'rf_importance': float(rf_importance[i]),
                'permutation_importance': float(perm_importance.importances_mean[i]),
                'correlation': float(correlations[i]) if not np.isnan(correlations[i]) else 0.0,
                'variance': float(normalized_variances[i]),
                'combined_score': float(
                    0.4 * rf_importance[i] + 
                    0.3 * perm_importance.importances_mean[i] + 
                    0.2 * abs(correlations[i]) if not np.isnan(correlations[i]) else 0.0 +
                    0.1 * normalized_variances[i]
                )
            }
        
        # Sort by combined score
        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        print(f"[FEATURES] Top 10 most important features:")
        for name, scores in sorted_features[:10]:
            print(f"  {name}: {scores['combined_score']:.4f}")
        
        return self.importance_scores
    
    def select_features(self, X: np.ndarray, top_k: int = 20) -> np.ndarray:
        """Select top K most important features"""
        
        if not self.importance_scores:
            raise ValueError("Run analyze_importance first")
        
        # Get top K features
        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        top_indices = [
            self.feature_names.index(name) 
            for name, _ in sorted_features[:top_k]
        ]
        
        return X[:, top_indices]
    
    def get_feature_interactions(self, X: np.ndarray) -> Dict:
        """Analyze feature interactions"""
        
        interactions = {}
        
        # Calculate pairwise correlations
        corr_matrix = np.corrcoef(X.T)
        
        for i in range(len(self.feature_names)):
            for j in range(i+1, len(self.feature_names)):
                if abs(corr_matrix[i, j]) > 0.7:  # Strong correlation
                    pair = f"{self.feature_names[i]}__{self.feature_names[j]}"
                    interactions[pair] = float(corr_matrix[i, j])
        
        return interactions
    
    def create_feature_engineering_suggestions(self, X: np.ndarray) -> List[str]:
        """Suggest new features based on analysis"""
        
        suggestions = []
        
        # Get top features
        sorted_features = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1]['combined_score'],
            reverse=True
        )
        
        top_features = [name for name, _ in sorted_features[:5]]
        
        # Suggest combinations
        for i, feat1 in enumerate(top_features):
            for j, feat2 in enumerate(top_features[i+1:], i+1):
                suggestions.append(f"ratio_{feat1}_{feat2}")
                suggestions.append(f"diff_{feat1}_{feat2}")
                suggestions.append(f"product_{feat1}_{feat2}")
        
        return suggestions[:10]  # Return top 10 suggestions
    
    def save_analysis(self, path: str = 'models/ml/feature_importance.json'):
        """Save feature importance analysis"""
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        output = {
            'importance_scores': self.importance_scores,
            'top_features': [
                name for name, _ in sorted(
                    self.importance_scores.items(),
                    key=lambda x: x[1]['combined_score'],
                    reverse=True
                )[:20]
            ],
            'feature_engineering_suggestions': self.create_feature_engineering_suggestions(None),
            'analysis_date': datetime.utcnow().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[FEATURES] Analysis saved to {path}")

# Main execution
if __name__ == "__main__":
    print("[FEATURES] Starting feature importance analysis...")
    
    # Generate synthetic data
    X = np.random.randn(1000, 43)
    y = np.random.randn(1000) * 10
    
    # Add some artificial relationships
    y += X[:, 0] * 2 + X[:, 5] * 1.5 + X[:, 10] * 0.8
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer()
    
    # Analyze importance
    importance = analyzer.analyze_importance(X, y)
    
    # Select top features
    X_selected = analyzer.select_features(X, top_k=20)
    print(f"[FEATURES] Selected shape: {X_selected.shape}")
    
    # Find interactions
    interactions = analyzer.get_feature_interactions(X)
    print(f"[FEATURES] Found {len(interactions)} strong feature interactions")
    
    # Get feature engineering suggestions
    suggestions = analyzer.create_feature_engineering_suggestions(X)
    print(f"[FEATURES] Generated {len(suggestions)} feature engineering suggestions")
    
    # Save analysis
    analyzer.save_analysis()
    print("[FEATURES] Analysis complete!")