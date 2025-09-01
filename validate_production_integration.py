#!/usr/bin/env python3
"""
Test script to verify all production ML models load correctly.
This validates the integration is working as expected.
"""

import numpy as np
import pandas as pd
import torch
import onnx
import onnxruntime as ort
from pathlib import Path
import sys

def test_onnx_model():
    """Test ONNX model loading and inference."""
    print("🔍 Testing ONNX model...")
    
    try:
        # Load ONNX model
        onnx_path = "models/rl_model.onnx"
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create inference session
        session = ort.InferenceSession(onnx_path)
        
        # Test inference
        input_name = session.get_inputs()[0].name
        test_input = np.random.randn(1, 30).astype(np.float32)
        
        output = session.run(None, {input_name: test_input})
        print(f"✅ ONNX model inference successful - output shape: {output[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX model test failed: {e}")
        return False

def test_pytorch_model():
    """Test PyTorch model loading."""
    print("🔍 Testing PyTorch model...")
    
    try:
        # Load checkpoint
        checkpoint = torch.load("models/rl_model.pth", map_location='cpu')
        
        # Verify checkpoint contents
        required_keys = ['model_state_dict', 'input_dim', 'hidden_dim', 'num_actions']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Missing required key: {key}")
        
        print(f"✅ PyTorch checkpoint valid - input_dim: {checkpoint['input_dim']}, actions: {checkpoint['num_actions']}")
        return True
        
    except Exception as e:
        print(f"❌ PyTorch model test failed: {e}")
        return False

def test_normalization_params():
    """Test normalization parameter loading."""
    print("🔍 Testing normalization parameters...")
    
    try:
        X_mean = np.load("models/rl_X_mean.npy")
        X_std = np.load("models/rl_X_std.npy")
        
        if X_mean.shape != X_std.shape:
            raise ValueError("Mean and std shapes don't match")
        
        if X_mean.shape[0] != 30:
            raise ValueError(f"Expected 30 features, got {X_mean.shape[0]}")
        
        print(f"✅ Normalization parameters valid - shape: {X_mean.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Normalization parameters test failed: {e}")
        return False

def test_training_data():
    """Test training data loading."""
    print("🔍 Testing training data...")
    
    try:
        df = pd.read_parquet("test_data.parquet")
        
        if len(df) == 0:
            raise ValueError("Training data is empty")
        
        required_columns = ['symbol', 'price', 'strategy', 'pnl']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        print(f"✅ Training data valid - {len(df)} rows, {len(df.columns)} columns")
        return True
        
    except Exception as e:
        print(f"❌ Training data test failed: {e}")
        return False

def test_cloud_learning_compatibility():
    """Test compatibility with cloud learning pipeline."""
    print("🔍 Testing cloud learning compatibility...")
    
    try:
        # Check if all required files exist
        required_files = [
            "models/rl_model.onnx",
            "models/rl_model.pth", 
            "models/rl_X_mean.npy",
            "models/rl_X_std.npy",
            "test_data.parquet"
        ]
        
        missing_files = [f for f in required_files if not Path(f).exists()]
        
        if missing_files:
            raise ValueError(f"Missing required files: {missing_files}")
        
        # Check GitHub Actions workflow exists
        workflow_path = ".github/workflows/train-continuous-final.yml"
        if not Path(workflow_path).exists():
            raise ValueError("GitHub Actions workflow not found")
        
        print("✅ Cloud learning compatibility verified")
        return True
        
    except Exception as e:
        print(f"❌ Cloud learning compatibility test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Production ML Integration Validation")
    print("=" * 50)
    
    tests = [
        test_onnx_model,
        test_pytorch_model,
        test_normalization_params,
        test_training_data,
        test_cloud_learning_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Production integration is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    # Install required packages if missing
    try:
        import onnxruntime
    except ImportError:
        print("Installing onnxruntime...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
        import onnxruntime as ort
    
    exit(main())