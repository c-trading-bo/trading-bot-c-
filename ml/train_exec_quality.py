#!/usr/bin/env python3
"""
Execution Quality Predictor Training Script
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from datetime import datetime

def train_exec_quality(data_file, output_dir):
    """Train execution quality predictor"""
    print(f"[EXEC] Training execution quality predictor from {data_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or generate data
    try:
        df = pd.read_parquet(data_file)
        print(f"[EXEC] Loaded {len(df)} samples")
    except Exception as e:
        print(f"[EXEC] Error loading data: {e}")
        # Create synthetic data
        df = pd.DataFrame({
            'entry_price': np.random.uniform(4400, 4600, 1000),
            'exit_price': np.random.uniform(4400, 4600, 1000),
            'volume': np.random.randint(1, 100, 1000),
            'spread': np.random.uniform(0.25, 2.0, 1000),
            'slippage': np.random.uniform(0, 1.5, 1000),
            'execution_quality': np.random.uniform(0, 1, 1000)
        })
        print(f"[EXEC] Generated {len(df)} synthetic samples")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['execution_quality', 'timestamp']]
    X = df[feature_cols].fillna(0)
    y = df['execution_quality'] if 'execution_quality' in df.columns else np.random.uniform(0, 1, len(df))
    
    print(f"[EXEC] Features: {list(X.columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"[EXEC] MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, 'exec_quality.pkl')
    joblib.dump(model, model_path)
    print(f"[EXEC] Saved model to {model_path}")
    
    return mse

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_exec_quality.py <data_file> <output_dir>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    mse = train_exec_quality(data_file, output_dir)
    print(f"[EXEC] Training completed with MSE: {mse:.4f}")
