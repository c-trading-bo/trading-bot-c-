# Neural-UCB Training Pipeline - Quick Start Guide

## Overview
The Neural-UCB training pipeline allows the bot to learn from Lab Mode trading data and retrain strategy selection models. This guide covers setup and validation for production deployment.

## Prerequisites

### 1. Python Installation
- **Required Version**: Python 3.8 or higher (tested with 3.12.3)
- **Installation Check**:
  ```bash
  python --version
  ```
- **Expected Output**: `Python 3.12.3` (or higher)

### 2. Required Python Packages
```bash
# Core dependencies
pip install torch numpy

# ONNX export dependencies (required for model generation)
pip install onnx onnxruntime onnxscript
```

### 3. Verify Installation
Run the quick test script:
```bash
python test-neural-ucb-pipeline.py
```

Expected output:
```
✅ ALL TESTS PASSED - Pipeline is ready!
```

## Usage

### Training from Lab Mode Data

The C# orchestrator automatically exports training data to:
```
models/neural_ucb_training_data.json
```

The training pipeline is automatically invoked during Lab Mode runs. Manual invocation:

```bash
python python/ucb/train_neural_ucb_from_strategy_data.py \
  --data-path models/neural_ucb_training_data.json
```

### Training Parameters

```bash
python python/ucb/train_neural_ucb_from_strategy_data.py \
  --data-path models/neural_ucb_training_data.json \
  --output-dir models \
  --checkpoint-path python/ucb/ucb_state.pkl \
  --input-dim 50 \
  --hidden-dim 128 \
  --learning-rate 0.001 \
  --batch-size 32 \
  --epochs 50
```

### Expected Outputs

After successful training, you should see:

1. **ONNX Models** (in `models/` directory):
   - `neural_ucb_model_S2.onnx` (~30KB)
   - `neural_ucb_model_S3.onnx` (~30KB)
   - `neural_ucb_model_S6.onnx` (~30KB)
   - `neural_ucb_model_S11.onnx` (~30KB)

2. **Python Checkpoint** (in `python/ucb/`):
   - `ucb_state.pkl` (~2MB)

3. **Console Output**:
   ```
   ✅ Training Complete!
   ONNX models saved: 4
   Checkpoint saved: python/ucb/ucb_state.pkl
   Models ready for C# inference via OnnxNeuralNetwork
   ```

## Validation

### Quick Validation Test
```bash
# Test with synthetic data
python python/ucb/train_neural_ucb_from_strategy_data.py \
  --data-path models/neural_ucb_training_data.json \
  --epochs 10

# Verify outputs
ls -lh models/neural_ucb_model_*.onnx
ls -lh python/ucb/ucb_state.pkl
```

### Full Validation Report
See: [NEURAL_UCB_PIPELINE_VALIDATION.md](NEURAL_UCB_PIPELINE_VALIDATION.md)

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Install PyTorch
```bash
pip install torch numpy
```

### Error: "ModuleNotFoundError: No module named 'onnxscript'"
**Solution**: Install ONNX dependencies
```bash
pip install onnx onnxruntime onnxscript
```

### Error: "FileNotFoundError: training data not found"
**Cause**: Training data file doesn't exist or path is incorrect

**Solution**: Verify the path to the JSON file:
```bash
ls -l models/neural_ucb_training_data.json
```

### Error: "python: command not found"
**Cause**: Python is not in PATH or not installed

**Solution 1**: Add Python to PATH (Linux/Mac)
```bash
export PATH="/usr/bin:$PATH"
```

**Solution 2**: Use full path in C# configuration
```csharp
// In appsettings.json or .env
"PythonExecutablePath": "/usr/bin/python3"
```

### Warning: "ONNX opset version 13 → 18"
**Impact**: None - This is informational. ONNX opset 18 is backward compatible.

**Optional Fix**: Update the training script to use opset 18:
```python
# In train_neural_ucb_from_strategy_data.py, line ~362
opset_version=18  # Change from 13 to 18
```

## Production Deployment Checklist

Before deploying to production:

- [ ] Python 3.8+ installed and accessible via `python` command
- [ ] PyTorch and NumPy installed (`pip list | grep torch`)
- [ ] ONNX libraries installed (`pip list | grep onnx`)
- [ ] Test script passes (`python test-neural-ucb-pipeline.py`)
- [ ] Training script runs without errors
- [ ] ONNX models generated successfully (4 files)
- [ ] Checkpoint file created (>1MB)
- [ ] C# can detect Python executable (`FindPythonExecutable()` succeeds)

## Architecture Overview

```
┌─────────────────────┐
│  Lab Mode Trading   │
│   (C# Bot Logic)    │
└─────────┬───────────┘
          │ Exports JSON
          ↓
┌─────────────────────┐
│ Training Data JSON  │
│ (Context + Rewards) │
└─────────┬───────────┘
          │ Invokes
          ↓
┌─────────────────────┐
│  Python Training    │
│  Script (PyTorch)   │
└─────────┬───────────┘
          │ Generates
          ↓
┌─────────────────────┐     ┌─────────────────────┐
│   ONNX Models       │     │  Python Checkpoint  │
│  (4 Strategy Arms)  │     │   (ucb_state.pkl)   │
└─────────┬───────────┘     └─────────────────────┘
          │ Loaded by
          ↓
┌─────────────────────┐
│  C# Model Inference │
│ (OnnxNeuralNetwork) │
└─────────────────────┘
```

## Files and Locations

| File | Location | Purpose | Size |
|------|----------|---------|------|
| Training Script | `python/ucb/train_neural_ucb_from_strategy_data.py` | Main training logic | ~20KB |
| Training Data | `models/neural_ucb_training_data.json` | Input data from Lab Mode | Varies |
| ONNX Models | `models/neural_ucb_model_{S2,S3,S6,S11}.onnx` | Trained models for C# | ~30KB each |
| Python Checkpoint | `python/ucb/ucb_state.pkl` | Python-side state | ~2MB |
| Test Script | `test-neural-ucb-pipeline.py` | Validation tool | ~4KB |

## Support

For issues or questions:
1. Check [NEURAL_UCB_PIPELINE_VALIDATION.md](NEURAL_UCB_PIPELINE_VALIDATION.md) for detailed validation results
2. Run `python test-neural-ucb-pipeline.py` to diagnose issues
3. Check log output from the training script for error messages

## Next Steps

After successful validation:
1. ✅ Pipeline is ready for Lab Mode runs
2. Monitor training logs during Lab Mode
3. Verify model files are generated after Lab Mode completes
4. Check model performance in subsequent trading sessions
