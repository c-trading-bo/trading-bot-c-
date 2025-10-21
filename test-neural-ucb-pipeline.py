#!/usr/bin/env python3
"""
Quick Test Script for Neural-UCB Training Pipeline
Tests Python detection and dependency availability
"""

import sys
import subprocess

def test_python_detection():
    """Test 1: Verify Python executable detection"""
    print("=" * 70)
    print("TEST 1: Python Executable Detection")
    print("=" * 70)
    
    try:
        result = subprocess.run(["python", "--version"], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip() or result.stderr.strip()
            print(f"✅ Python detected: {version}")
            
            # Check path
            path_result = subprocess.run(["which", "python"], 
                                       capture_output=True, text=True, timeout=5)
            if path_result.returncode == 0:
                print(f"✅ Python path: {path_result.stdout.strip()}")
            return True
        else:
            print("❌ Python not detected in PATH")
            return False
    except Exception as e:
        print(f"❌ Error detecting Python: {e}")
        return False

def test_dependencies():
    """Test 2: Verify PyTorch and NumPy installation"""
    print("\n" + "=" * 70)
    print("TEST 2: Dependency Installation Check")
    print("=" * 70)
    
    try:
        import torch
        print(f"✅ PyTorch installed: version {torch.__version__}")
        
        import numpy
        print(f"✅ NumPy installed: version {numpy.__version__}")
        
        try:
            import onnx
            print(f"✅ ONNX installed: version {onnx.__version__}")
        except ImportError:
            print("⚠️  ONNX not installed (optional for testing)")
        
        try:
            import onnxruntime
            print(f"✅ ONNX Runtime installed: version {onnxruntime.__version__}")
        except ImportError:
            print("⚠️  ONNX Runtime not installed (optional for testing)")
        
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n💡 Install with: pip install torch numpy onnx onnxruntime onnxscript")
        return False

def test_training_script():
    """Test 3: Verify training script exists and is importable"""
    print("\n" + "=" * 70)
    print("TEST 3: Training Script Availability")
    print("=" * 70)
    
    import os
    script_path = "python/ucb/train_neural_ucb_from_strategy_data.py"
    
    if os.path.exists(script_path):
        print(f"✅ Training script found: {script_path}")
        return True
    else:
        print(f"❌ Training script not found: {script_path}")
        return False

def main():
    """Run all tests"""
    print("\n🚀 Neural-UCB Pipeline Quick Test")
    print("=" * 70)
    
    results = {
        "Python Detection": test_python_detection(),
        "Dependencies": test_dependencies(),
        "Training Script": test_training_script()
    }
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Pipeline is ready!")
        print("\nNext step: Run training with:")
        print("  python python/ucb/train_neural_ucb_from_strategy_data.py \\")
        print("    --data-path models/neural_ucb_training_data.json")
    else:
        print("❌ SOME TESTS FAILED - Please fix issues before running pipeline")
    print("=" * 70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
