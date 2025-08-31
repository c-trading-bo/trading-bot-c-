#!/usr/bin/env python3
"""
Train meta strategy classifier for 24/7 cloud learning pipeline.
Determines which strategy to use based on market conditions.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os
import sys

def train_meta_classifier(data_file='../data/logs/candidates.merged.parquet', output_dir='models'):
    """Train meta strategy classifier and save as ONNX."""
    
    # Load merged data
    df = pd.read_parquet(data_file)
    print(f'📊 Training meta classifier on {len(df)} samples')

    # Feature engineering for meta strategy selection
    feature_cols = [
        'price', 'atr', 'rsi', 'ema20', 'ema50', 'volume', 'spread', 
        'volatility', 'signal_strength', 'prior_win_rate', 'avg_r_multiple'
    ]

    # Handle missing columns gracefully
    available_features = [col for col in feature_cols if col in df.columns]
    print(f'Available features: {available_features}')

    if not available_features:
        print('ERROR: No valid features found in dataset!')
        sys.exit(1)

    X = df[available_features].fillna(0)
    
    if 'strategy' not in df.columns:
        print('ERROR: No strategy column found!')
        sys.exit(1)
        
    y = df['strategy']

    # Encode strategy labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f'Strategy classes: {le.classes_}')
    print(f'Class distribution: {pd.Series(y).value_counts().to_dict()}')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train meta classifier
    meta_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    meta_model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = meta_model.predict(X_test_scaled)
    print('📊 Meta Strategy Classifier Performance:')
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': meta_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print('\n📈 Feature Importance:')
    print(feature_importance.head(10).to_string(index=False))

    # Save models
    os.makedirs(output_dir, exist_ok=True)
    pickle.dump(scaler, open(f'{output_dir}/meta_scaler.pkl', 'wb'))
    pickle.dump(le, open(f'{output_dir}/meta_encoder.pkl', 'wb'))

    # Convert to ONNX
    output_file = f'{output_dir}/meta_model.onnx'
    initial_type = [('float_input', FloatTensorType([None, len(available_features)]))]
    
    try:
        meta_onnx = convert_sklearn(meta_model, initial_types=initial_type)
        with open(output_file, 'wb') as f:
            f.write(meta_onnx.SerializeToString())
        print(f'✅ Meta classifier saved to {output_file}')
    except Exception as e:
        print(f'ERROR converting to ONNX: {e}')
        # Save as pickle fallback
        pickle.dump(meta_model, open(f'{output_dir}/meta_model.pkl', 'wb'))
        print(f'✅ Meta classifier saved as pickle to {output_dir}/meta_model.pkl')

    return meta_model, scaler, le

if __name__ == '__main__':
    data_file = sys.argv[1] if len(sys.argv) > 1 else '../data/logs/candidates.merged.parquet'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'models'
    train_meta_classifier(data_file, output_dir)
