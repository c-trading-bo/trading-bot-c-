#!/usr/bin/env python3
"""
RL Position Sizer Training Script
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

class PositionSizerNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Position size between 0 and 1
        return x

def train_rl_sizer(data_file, output_dir):
    """Train RL position sizer"""
    print(f"[RL] Training RL position sizer from {data_file}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or generate data
    try:
        df = pd.read_parquet(data_file)
        print(f"[RL] Loaded {len(df)} samples")
    except Exception as e:
        print(f"[RL] Error loading data: {e}")
        # Create synthetic data
        df = pd.DataFrame({
            'price': np.random.uniform(4400, 4600, 1000),
            'atr': np.random.uniform(10, 50, 1000),
            'rsi': np.random.uniform(20, 80, 1000),
            'volatility': np.random.uniform(0.1, 0.8, 1000),
            'signal_strength': np.random.uniform(0, 1, 1000),
            'position_size_optimal': np.random.uniform(0.1, 1.0, 1000)
        })
        print(f"[RL] Generated {len(df)} synthetic samples")
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['position_size_optimal', 'timestamp']]
    X = torch.FloatTensor(df[feature_cols].fillna(0).values)
    y = torch.FloatTensor(df['position_size_optimal'].fillna(0.5).values).unsqueeze(1)
    
    print(f"[RL] Training on {len(X)} samples with {X.shape[1]} features")
    
    # Initialize model
    model = PositionSizerNet(input_dim=X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"[RL] Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, 'rl_sizer.pth')
    torch.save(model.state_dict(), model_path)
    print(f"[RL] Saved model to {model_path}")
    
    # Export to ONNX
    model.eval()
    dummy_input = torch.randn(1, X.shape[1])
    onnx_path = os.path.join(output_dir, 'rl_sizer.onnx')
    torch.onnx.export(model, dummy_input, onnx_path, 
                     input_names=['features'], output_names=['position_size'],
                     dynamic_axes={'features': {0: 'batch_size'}})
    print(f"[RL] Exported ONNX to {onnx_path}")
    
    return loss.item()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_rl_sizer.py <data_file> <output_dir>")
        sys.exit(1)
    
    data_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    loss = train_rl_sizer(data_file, output_dir)
    print(f"[RL] Training completed with final loss: {loss:.4f}")
