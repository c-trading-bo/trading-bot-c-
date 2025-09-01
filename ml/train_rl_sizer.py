"""
Train RL position sizer from merged training data.
Usage: python train_rl_sizer.py <data_file> <models_dir>
"""
import sys
import os
import pandas as pd
import torch
import torch.nn as nn

class PositionSizer(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1, scale to 0.1-2.0
        )
    
    def forward(self, x):
        return self.network(x) * 1.9 + 0.1  # Scale to 0.1-2.0 range

def main():
    if len(sys.argv) < 3:
        print("Usage: python train_rl_sizer.py <data_file> <models_dir>")
        sys.exit(1)
        
    data_file = sys.argv[1]
    models_dir = sys.argv[2]
    
    # Load data
    df = pd.read_parquet(data_file)
    print(f'📊 Training RL position sizer on {len(df)} samples')

    # Display symbol distribution if available
    if 'symbol' in df.columns:
        symbol_dist = df['symbol'].value_counts()
        print(f'📊 Symbol distribution: {symbol_dist.to_dict()}')
        
        # Add symbol-specific features
        df['is_es'] = (df['symbol'] == 'ES').astype(float)
        df['is_nq'] = (df['symbol'] == 'NQ').astype(float)
    else:
        # Fallback if no symbol column
        df['is_es'] = 1.0
        df['is_nq'] = 0.0

    # Prepare features (now symbol-aware)
    feature_cols = ['price', 'atr', 'rsi', 'volume', 'signal_strength', 'prior_win_rate', 'avg_r_multiple', 'drawdown_risk', 'is_es', 'is_nq']
    available_features = [col for col in feature_cols if col in df.columns]

    if not available_features:
        available_features = ['price', 'atr', 'rsi', 'is_es', 'is_nq']  # Fallback with symbol features

    X = df[available_features].fillna(0).values.astype('float32')
    y = df.get('r_multiple', pd.Series([0.1] * len(df))).fillna(0).values.astype('float32')

    # Normalize features
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0) + 1e-8
    x_norm = (X - x_mean) / x_std

    # Simple training
    model = PositionSizer(len(available_features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Add weight_decay
    criterion = nn.MSELoss()

    x_tensor = torch.FloatTensor(x_norm)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)

    print('🤖 Training RL model...')
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{models_dir}/rl_model.pth')

    # Save normalization parameters
    import numpy as np
    np.save(f'{models_dir}/rl_X_mean.npy', x_mean)
    np.save(f'{models_dir}/rl_X_std.npy', x_std)

    # Export to ONNX
    dummy_input = torch.randn(1, len(available_features))
    torch.onnx.export(model, (dummy_input,), f'{models_dir}/rl_model.onnx',  # Pass as tuple
                     input_names=['observation'], output_names=['position_size'],
                     dynamic_axes={'observation': {0: 'batch_size'}})

    print('✅ RL position sizer saved to rl_model.onnx')

if __name__ == "__main__":
    main()
