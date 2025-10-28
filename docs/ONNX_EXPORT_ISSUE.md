# ONNX Export Issue Documentation

## Problem Statement

The neural network trainers were originally designed to export models in ONNX format for production inference. However, TorchSharp's ONNX export functionality has compatibility issues that prevent reliable model serialization.

## Current State

### Training (Lab Mode)
- **Format**: TorchSharp native JSON format (`.json` files)
- **Method**: `Module.save(path)` and `Module.load(path)`
- **Location**: `models/{trainer}/v{version}_{timestamp}/`
- **Status**: ✅ Working - All 7 trainers successfully save/load

### Inference (Production)
- **Format**: ONNX (`.onnx` files)
- **Method**: `OnnxModelLoader` and `InferenceSession`
- **Location**: Various model paths in UnifiedTradingBrain
- **Status**: ❌ Mismatch - Expects ONNX but trainers produce TorchSharp JSON

## Root Cause

TorchSharp's ONNX export (`torch.onnx.export()`) has known limitations:
1. Not all TorchSharp operations are supported in ONNX opset
2. Custom layer implementations may not convert correctly
3. Dynamic shapes and control flow can cause export failures
4. BatchNorm and Dropout layers may have compatibility issues

## Impact

Currently trained models cannot be directly loaded for production inference because:
- Trainers save in TorchSharp JSON format (only option that works reliably)
- UnifiedTradingBrain expects ONNX format for inference
- No automatic conversion exists between the formats

## Solutions (Not Implemented)

### Option 1: TorchSharp Inference (Recommended)
**Pros**:
- Uses native format from training
- No conversion needed
- Better compatibility

**Cons**:
- Requires changing UnifiedTradingBrain inference code
- May have different performance characteristics than ONNX Runtime

**Implementation**:
1. Add TorchSharp inference paths to UnifiedTradingBrain
2. Load models using `Module.load()` instead of ONNX InferenceSession
3. Wrap TorchSharp inference in the existing model interface

### Option 2: Manual ONNX Conversion
**Pros**:
- Minimal changes to production inference code
- ONNX Runtime optimizations available

**Cons**:
- Requires reliable ONNX export (currently broken)
- Extra conversion step after training
- May lose model fidelity in conversion

**Implementation**:
1. Fix TorchSharp ONNX export issues (complex, may not be possible)
2. Add post-training conversion step to orchestrator
3. Validate converted models match original accuracy

### Option 3: Dual Format Support
**Pros**:
- Flexibility to use either format
- Gradual migration path

**Cons**:
- More complex code paths
- Increased maintenance burden

## Recommended Next Steps

1. **Short-term**: Document this limitation clearly
2. **Medium-term**: Implement Option 1 (TorchSharp inference)
3. **Long-term**: Investigate ONNX export fixes with TorchSharp team

## Files Affected

### Training (TorchSharp JSON)
- `src/RLAgent/LSTMTrainer.cs`
- `src/RLAgent/Algorithms/SACTrainer.cs`
- `src/BotCore/Bandits/NeuralUcbBanditTrainer.cs`
- `src/RLAgent/PatternRecognitionTrainer.cs`
- `src/RLAgent/RegimeDetectorTrainer.cs`
- `src/RLAgent/ModelEnsembleTrainer.cs`
- `src/RLAgent/SlippageLatencyTrainer.cs`

### Inference (ONNX)
- `src/BotCore/Brain/UnifiedTradingBrain.cs`
- `src/BotCore/ML/OnnxModelLoader.cs`

## Workaround (Temporary)

Until a proper solution is implemented:
- CVaRPPOTrainer already has working SaveModelAsync that saves TorchSharp format
- Other trainers now also save in TorchSharp format
- Production inference continues to use existing ONNX models (outdated)
- New trained models are not automatically deployed to production

## Related Issues

- Step 9: Update UnifiedTradingBrain to load JSON instead of ONNX
- Step 10: Test all models load correctly
- Step 11: Verify inference works with loaded models
- Step 15: Delete 108 broken ONNX stub files (optional cleanup)

## Date
2025-10-28

## Author
Copilot AI Agent (via model persistence implementation PR)
