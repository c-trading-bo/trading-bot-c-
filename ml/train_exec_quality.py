"""
Train execution quality predictor from merged training data.
Usage: python train_exec_quality.py <data_file> <models_dir>
"""
import sys
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: skl2onnx not available, will save as pickle")

def main():
    if len(sys.argv) < 3:
        print("Usage: python train_exec_quality.py <data_file> <models_dir>")
        sys.exit(1)
        
    data_file = sys.argv[1]
    models_dir = sys.argv[2]
    
    # Load merged data
    df = pd.read_parquet(data_file)
    print(f'📊 Training execution predictor on {len(df)} samples')

    # Features for execution quality prediction
    feature_cols = [
        'price', 'atr', 'rsi', 'volume', 'spread', 'volatility',
        'bid_ask_imbalance', 'order_book_imbalance', 'tick_direction',
        'signal_strength', 'liquidity_risk'
    ]

    available_features = [col for col in feature_cols if col in df.columns]
    print(f'Available execution features: {available_features}')

    if not available_features:
        print("No features available for training")
        sys.exit(1)

    X = df[available_features].fillna(0)
    y = df.get('r_multiple', pd.Series([0.1] * len(df))).fillna(0)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled = scaler.transform(X_test)

    # Train execution quality model
    exec_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6)
    exec_model.fit(x_train_scaled, y_train)

    # Evaluate
    y_pred = exec_model.predict(x_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('📊 Execution Quality Predictor Performance:')  # Remove f-string formatting
    print(f'MSE: {mse:.4f}, R²: {r2:.4f}')

    # Save models
    os.makedirs(models_dir, exist_ok=True)
    pickle.dump(scaler, open(f'{models_dir}/exec_scaler.pkl', 'wb'))

    # Convert to ONNX
    if ONNX_AVAILABLE:
        try:
            initial_type = [('float_input', FloatTensorType([None, len(available_features)]))]
            exec_onnx = convert_sklearn(exec_model, initial_types=initial_type)
            with open(f'{models_dir}/exec_model.onnx', 'wb') as f:
                # convert_sklearn returns a tuple (ModelProto, Topology), we want the ModelProto
                if isinstance(exec_onnx, tuple):
                    model_proto = exec_onnx[0]
                else:
                    model_proto = exec_onnx
                f.write(model_proto.SerializeToString())
            print('✅ Execution predictor saved to exec_model.onnx')
        except Exception as e:
            print(f"ONNX conversion failed: {e}")
            pickle.dump(exec_model, open(f'{models_dir}/exec_model.pkl', 'wb'))
            print('✅ Execution predictor saved as pickle')
    else:
        pickle.dump(exec_model, open(f'{models_dir}/exec_model.pkl', 'wb'))
        print('✅ Execution predictor saved as pickle')

if __name__ == "__main__":
    main()
