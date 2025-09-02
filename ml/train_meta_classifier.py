#!/usr/bin/env python3
"""
Meta Strategy Classifier Training Script
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import onnx
from skl2onnx import to_onnx
from datetime import datetime

def train_meta_classifier(data_file, output_dir):
    """Train meta strategy classifier"""
    print(f"[META] Training meta classifier from {data_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_parquet(data_file)
        print(f"[META] Loaded {len(df)} samples")
    except Exception as e:
        print(f"[META] Error loading data: {e}")
        # Create synthetic data
        df = pd.DataFrame({
            'price': np.random.uniform(4400, 4600, 1000),
            'atr': np.random.uniform(10, 50, 1000),
            'rsi': np.random.uniform(20, 80, 1000),
            'r_multiple': np.random.uniform(-3, 5, 1000),
            'win': np.random.choice([True, False], 1000)
        })
        print(f"[META] Generated {len(df)} synthetic samples")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['win', 'r_multiple', 'timestamp', 'symbol']]
    X = df[feature_cols].fillna(0)
    y = df['win'].astype(int) if 'win' in df.columns else np.random.choice([0, 1], len(df))
    
    print(f"[META] Features: {list(X.columns)}")
    print(f"[META] Target distribution: {np.bincount(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    print(f"[META] Random Forest Accuracy: {rf_accuracy:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=5)
    print(f"[META] CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Save models
    model_path = os.path.join(output_dir, 'meta_classifier.pkl')
    joblib.dump(rf_model, model_path)
    print(f"[META] Saved model to {model_path}")
    
    # Export to ONNX
    try:
        onnx_model = to_onnx(rf_model, X_train.values.astype(np.float32))
        onnx_path = os.path.join(output_dir, 'meta_classifier.onnx')
        with open(onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        print(f"[META] Exported ONNX to {onnx_path}")
    except Exception as e:
        print(f"[META] ONNX export failed: {e}")
    
    return rf_accuracy

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_meta_classifier.py <data_file> <output_dir>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    accuracy = train_meta_classifier(data_file, output_dir)
    print(f"[META] Training completed with accuracy: {accuracy:.3f}")
