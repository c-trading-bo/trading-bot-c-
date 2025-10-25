"""
Gradient Boosting Model Trainer
Trains XGBoost and LightGBM models to complement deep learning ensemble
Addresses HEDGE_FUND_GAP_ANALYSIS.md - Section 3: XGBoost Ensemble
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")


class GradientBoostingTrainer:
    """
    Trains gradient boosting models (XGBoost/LightGBM) on historical trading data
    Complements existing deep learning models with tree-based ensemble
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to training configuration JSON
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.symbol = self.config.get('symbol', 'ES')
        self.model_type = self.config.get('modelType', 'xgboost')
        self.hyperparameters = self.config.get('hyperparameters', {})
        self.output_path = self.config.get('outputPath', './models/gradient_boosting')
        
        # Create output directory
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized GradientBoostingTrainer for {self.symbol} using {self.model_type}")
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load historical data for training
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        # Look for historical data in datasets directory
        data_path = f"./datasets/{self.symbol}_90days.json"
        
        if not os.path.exists(data_path):
            print(f"Warning: Historical data not found at {data_path}")
            # Generate synthetic data for demonstration
            return self._generate_synthetic_data()
        
        print(f"Loading data from {data_path}")
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Engineer features
        features = self._engineer_features(df)
        
        # Create target (1 = bullish, 0 = bearish)
        target = self._create_target(df)
        
        return features, target
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer trading features from OHLCV data
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Feature DataFrame
        """
        features = pd.DataFrame()
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['high_low_range'] = (df['high'] - df['low']) / df['close']
        features['close_open_range'] = (df['close'] - df['open']) / df['open']
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'ma_{period}'] = df['close'].rolling(period).mean() / df['close']
        
        # Momentum indicators
        features['rsi'] = self._calculate_rsi(df['close'], 14)
        features['macd'] = self._calculate_macd(df['close'])
        
        # Volatility
        features['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # Drop NaN rows
        features = features.dropna()
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> pd.Series:
        """Calculate MACD indicator"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd / prices
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary target: 1 if next period is bullish, 0 otherwise
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Target Series
        """
        # Target: positive return in next period
        future_returns = df['close'].shift(-1) / df['close'] - 1
        target = (future_returns > 0).astype(int)
        return target[:-1]  # Drop last row (no future data)
    
    def _generate_synthetic_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate synthetic data for demonstration"""
        print("Generating synthetic training data...")
        
        n_samples = 1000
        n_features = 15
        
        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        target = pd.Series(np.random.randint(0, 2, n_samples))
        
        return features, target
    
    def train_xgboost(self, X_train, y_train, X_val, y_val) -> Tuple[Any, Dict]:
        """
        Train XGBoost model
        
        Returns:
            Tuple of (model, metrics)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")
        
        print("Training XGBoost model...")
        
        # Default hyperparameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42
        }
        
        # Override with config
        params.update(self.hyperparameters)
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate metrics
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        print(f"XGBoost Metrics - Accuracy: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}, AUC: {metrics['auc']:.4f}")
        
        return model, metrics
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val) -> Tuple[Any, Dict]:
        """
        Train LightGBM model
        
        Returns:
            Tuple of (model, metrics)
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        print("Training LightGBM model...")
        
        # Default hyperparameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary',
            'metric': 'auc',
            'random_state': 42,
            'verbose': -1
        }
        
        # Override with config
        params.update(self.hyperparameters)
        
        # Train model
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Calculate metrics
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'auc': roc_auc_score(y_val, y_pred_proba)
        }
        
        print(f"LightGBM Metrics - Accuracy: {metrics['accuracy']:.4f}, "
              f"F1: {metrics['f1_score']:.4f}, AUC: {metrics['auc']:.4f}")
        
        return model, metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Execute training pipeline
        
        Returns:
            Training results dictionary
        """
        # Load data
        features, target = self.load_training_data()
        
        # Align features and target
        min_len = min(len(features), len(target))
        features = features.iloc[:min_len]
        target = target.iloc[:min_len]
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Train model
        if self.model_type == 'xgboost':
            model, metrics = self.train_xgboost(X_train, y_train, X_val, y_val)
        elif self.model_type == 'lightgbm':
            model, metrics = self.train_lightgbm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Save model
        model_id = f"{self.model_type}_{self.symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        model_path = os.path.join(self.output_path, f"{model_id}.json")
        
        # Save in JSON format
        if self.model_type == 'xgboost':
            model.save_model(model_path)
        elif self.model_type == 'lightgbm':
            model.booster_.save_model(model_path)
        
        print(f"Model saved to: {model_path}")
        
        # Save metrics
        results = {
            'model_id': model_id,
            'model_type': self.model_type,
            'symbol': self.symbol,
            'training_date': datetime.utcnow().isoformat(),
            'metrics': metrics,
            'model_path': model_path,
            'feature_names': list(features.columns)
        }
        
        results_path = os.path.join(self.output_path, f"{model_id}_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")
        
        return results


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python gradient_boosting_trainer.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    trainer = GradientBoostingTrainer(config_path)
    results = trainer.train()
    
    print("\n=== Training Complete ===")
    print(f"Model ID: {results['model_id']}")
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
    print(f"AUC: {results['metrics']['auc']:.4f}")


if __name__ == '__main__':
    main()
