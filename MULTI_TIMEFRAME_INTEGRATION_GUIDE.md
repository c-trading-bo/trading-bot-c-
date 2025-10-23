# Multi-Timeframe Training Integration Guide

## Overview

This guide explains how to use the multi-timeframe coordinated training infrastructure for training ML models with synchronized 5-minute and 1-minute bar data.

**Status**: ✅ Infrastructure Components Complete  
**Next Step**: Multi-Branch Model Implementation (Future Work)

---

## Architecture Components

### 1. Data Loading Layer
**Component**: `MultiTimeframeDataLoader`  
**Purpose**: Loads historical 5m and 1m bars from JSON files  
**Location**: `src/BotCore/ML/MultiTimeframeDataLoader.cs`

```csharp
// Load synchronized historical data
var dataLoader = serviceProvider.GetRequiredService<MultiTimeframeDataLoader>();
var (bars5m, bars1m) = dataLoader.LoadHistoricalData("ES");
```

**Input**: JSON files in `data/historical/`
- `ES_90days.json` - 5-minute bars
- `ES_1m_90days.json` - 1-minute bars

**Output**: Chronologically sorted bar lists

### 2. Feature Extraction Layer
**Component**: `MultiTimeframeFeatureExtractor`  
**Purpose**: Computes technical indicators from both timeframes  
**Location**: `src/BotCore/ML/MultiTimeframeFeatureExtractor.cs`

```csharp
// Extract features from bars
var extractor = serviceProvider.GetRequiredService<MultiTimeframeFeatureExtractor>();

// 5m features: ATR, RSI, MACD, volume imbalance, trend slope
var features5m = extractor.Extract5mFeatures(bars5m);

// 1m features: Same indicators but faster windows
var features1m = extractor.Extract1mFeatures(bars1m);

// Synchronized features at a specific timestamp
var syncedFeatures = extractor.SynchronizeFeatures(timestamp, bars5m, bars1m);
```

**Features Extracted**:
- **5-minute timeframe** (strategic context):
  - ATR (14-period window)
  - RSI (14-period)
  - MACD (12/26/9)
  - Volume imbalance (20-period)
  - Trend slope (10-period linear regression)

- **1-minute timeframe** (tactical context):
  - ATR (14-period, more responsive)
  - RSI (14-period)
  - MACD (5/13/5, faster periods)
  - Volume imbalance (20-period)
  - Trend slope (10-period)

### 3. Data Assembly Layer
**Component**: `MultiTimeframeDataAssembler`  
**Purpose**: Creates synchronized training samples  
**Location**: `src/BotCore/ML/MultiTimeframeDataAssembler.cs`

```csharp
// Assemble synchronized samples
var assembler = serviceProvider.GetRequiredService<MultiTimeframeDataAssembler>();
var samples = assembler.AssembleSamples("ES", bars5m, bars1m);

// Each sample contains:
// - 36 bars of 5m context (3 hours strategic)
// - 60 bars of 1m context (1 hour tactical)
// - Features from both timeframes
// - Label (forward-looking outcome)
```

**Sample Structure**:
```csharp
public class EnhancedMultiTimeframeSample
{
    public string Symbol { get; set; }
    public DateTimeOffset Timestamp { get; set; }
    
    // Raw bar context
    public List<BarData> Context5m { get; set; }  // 36 bars
    public List<BarData> Context1m { get; set; }  // 60 bars
    
    // Extracted features
    public Dictionary<string, double> Features5m { get; set; }
    public Dictionary<string, double> Features1m { get; set; }
    
    // Supervised learning label
    public double Label { get; set; }  // 1.0 = up, -1.0 = down, 0.0 = flat
}
```

### 4. Batch Creation Layer
**Component**: `MultiTimeframeBatchCreator`  
**Purpose**: Groups samples into batches for efficient GPU training  
**Location**: `src/BotCore/ML/MultiTimeframeBatchCreator.cs`

```csharp
// Create batches for training
var batchCreator = serviceProvider.GetRequiredService<MultiTimeframeBatchCreator>();
var batches = batchCreator.CreateBatches(
    samples, 
    batchSize: 32, 
    shuffle: true
);

// Each batch contains aligned arrays:
// - Features5m: [batch_size, num_features_5m]
// - Features1m: [batch_size, num_features_1m]
// - Labels: [batch_size]
// - Masks: [batch_size, sequence_length] (for future sequence models)
```

**Batch Structure**:
```csharp
public class MultiTimeframeBatch
{
    public int BatchSize { get; set; }
    public List<string> Symbols { get; set; }
    public List<DateTimeOffset> Timestamps { get; set; }
    
    // Feature arrays (ready for neural networks)
    public double[,] Features5m { get; set; }     // [batch_size, num_features_5m]
    public double[,] Features1m { get; set; }     // [batch_size, num_features_1m]
    
    // Attention masks (for sequence models)
    public int[,] Mask5m { get; set; }
    public int[,] Mask1m { get; set; }
    
    // Labels for supervised learning
    public double[] Labels { get; set; }
}
```

### 5. Complete Pipeline
**Component**: `MultiTimeframeTrainingPipeline`  
**Purpose**: Orchestrates all components together  
**Location**: `src/BotCore/ML/MultiTimeframeTrainingPipeline.cs`

```csharp
// Use the complete pipeline
var pipeline = serviceProvider.GetRequiredService<MultiTimeframeTrainingPipeline>();

var trainingData = await pipeline.PrepareTrainingDataAsync(
    symbol: "ES",
    trainRatio: 0.67,   // 67% for training
    valRatio: 0.17,     // 17% for validation
    batchSize: 32,
    shuffle: true
);

// Access prepared batches
var trainBatches = trainingData.TrainBatches;         // Ready for training
var valBatches = trainingData.ValidationBatches;       // For hyperparameter tuning
var testBatches = trainingData.TestBatches;           // For final evaluation

// Dataset statistics
Console.WriteLine($"Total samples: {trainingData.Statistics.TotalSamples}");
Console.WriteLine($"Train: {trainingData.Statistics.TrainSamples}");
Console.WriteLine($"Val: {trainingData.Statistics.ValidationSamples}");
Console.WriteLine($"Test: {trainingData.Statistics.TestSamples}");
Console.WriteLine($"Features: {trainingData.Statistics.TotalFeatures}");
```

---

## Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                     MULTI-TIMEFRAME PIPELINE                      │
└──────────────────────────────────────────────────────────────────┘

1. DATA LOADING
   ┌─────────────────┐
   │ JSON Files      │
   │ - ES_90days.json│  ──→  MultiTimeframeDataLoader
   │ - ES_1m_90days  │
   └─────────────────┘
         │
         ├─→ bars5m (List<BarData>) - 5-minute bars
         └─→ bars1m (List<BarData>) - 1-minute bars

2. FEATURE EXTRACTION
   bars5m + bars1m  ──→  MultiTimeframeFeatureExtractor
         │
         ├─→ features5m (ATR, RSI, MACD, Volume, Trend)
         └─→ features1m (ATR, RSI, MACD, Volume, Trend)

3. DATA ASSEMBLY
   bars5m + bars1m + features  ──→  MultiTimeframeDataAssembler
         │
         └─→ samples (List<EnhancedMultiTimeframeSample>)
             Each sample:
             - Timestamp (decision point)
             - Context5m (36 bars)
             - Context1m (60 bars)
             - Features5m (dict)
             - Features1m (dict)
             - Label (1/-1/0)

4. TRAIN/VAL/TEST SPLIT
   samples  ──→  Chronological Split (67%/17%/16%)
         │
         ├─→ trainSamples (oldest)
         ├─→ valSamples (middle)
         └─→ testSamples (newest)

5. BATCH CREATION
   trainSamples  ──→  MultiTimeframeBatchCreator
         │
         └─→ batches (List<MultiTimeframeBatch>)
             Each batch:
             - Features5m: [32, num_features_5m]
             - Features1m: [32, num_features_1m]
             - Labels: [32]

6. MODEL TRAINING (FUTURE WORK)
   batches  ──→  Multi-Branch Neural Network
         │
         ├─→ 5m Branch (processes Features5m)
         ├─→ 1m Branch (processes Features1m)
         └─→ Fusion Layer (combines both branches)
              │
              └─→ Predictions
```

---

## Usage Examples

### Example 1: Basic Pipeline Usage

```csharp
using Microsoft.Extensions.DependencyInjection;
using BotCore.ML;

public class TrainingExample
{
    private readonly MultiTimeframeTrainingPipeline _pipeline;
    
    public TrainingExample(IServiceProvider serviceProvider)
    {
        _pipeline = serviceProvider.GetRequiredService<MultiTimeframeTrainingPipeline>();
    }
    
    public async Task RunTrainingAsync()
    {
        // Prepare data
        var data = await _pipeline.PrepareTrainingDataAsync("ES");
        
        // Iterate through training batches
        foreach (var batch in data.TrainBatches)
        {
            // batch.Features5m contains 5-minute features
            // batch.Features1m contains 1-minute features
            // batch.Labels contains labels
            
            // TODO: Pass to multi-branch model for training
            // await model.TrainAsync(batch.Features5m, batch.Features1m, batch.Labels);
        }
        
        // Validate on validation set
        foreach (var batch in data.ValidationBatches)
        {
            // TODO: Run validation
            // var valLoss = await model.ValidateAsync(batch.Features5m, batch.Features1m, batch.Labels);
        }
        
        // Final test on test set
        foreach (var batch in data.TestBatches)
        {
            // TODO: Run test
            // var testMetrics = await model.TestAsync(batch.Features5m, batch.Features1m, batch.Labels);
        }
    }
}
```

### Example 2: Custom Batch Processing

```csharp
public class CustomBatchProcessor
{
    private readonly MultiTimeframeTrainingPipeline _pipeline;
    
    public async Task ProcessCustomBatchesAsync()
    {
        var data = await _pipeline.PrepareTrainingDataAsync(
            symbol: "NQ",
            trainRatio: 0.70,
            valRatio: 0.15,
            batchSize: 64,  // Larger batches for GPU
            shuffle: true
        );
        
        // Access raw samples if you need custom batching
        var trainSamples = data.TrainSamples;
        
        // Process each sample individually
        foreach (var sample in trainSamples)
        {
            // Access raw bar context
            var bars5m = sample.Context5m;  // 36 bars
            var bars1m = sample.Context1m;  // 60 bars
            
            // Access extracted features
            var atr5m = sample.Features5m["atr_5m"];
            var rsi1m = sample.Features1m["rsi_1m"];
            
            // Access label
            var label = sample.Label;  // 1.0, -1.0, or 0.0
            
            // Custom processing...
        }
    }
}
```

### Example 3: Integration with Existing Trainers

```csharp
public class MultiTimeframeIntegrationExample
{
    private readonly MultiTimeframeTrainingPipeline _pipeline;
    private readonly CVaRPPOTrainer _cvarTrainer;
    
    public async Task IntegrateWithCVaRPPOAsync()
    {
        // Step 1: Prepare multi-timeframe data
        var data = await _pipeline.PrepareTrainingDataAsync("ES");
        
        // Step 2: Convert to format expected by existing trainer
        var experiences = ConvertToExperiences(data.TrainSamples);
        
        // Step 3: Train using existing trainer
        // NOTE: This is a placeholder - full integration requires
        // modifying CVaRPPOTrainer to accept multi-timeframe features
        // await _cvarTrainer.TrainAsync(experiences);
    }
    
    private List<Experience> ConvertToExperiences(
        List<EnhancedMultiTimeframeSample> samples)
    {
        // Convert multi-timeframe samples to Experience format
        // This is where you'd map features to the format expected
        // by the RL trainer
        
        var experiences = new List<Experience>();
        foreach (var sample in samples)
        {
            // Combine features from both timeframes
            var allFeatures = sample.Features5m
                .Concat(sample.Features1m)
                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
            
            var experience = new Experience
            {
                State = ConvertFeaturesToState(allFeatures),
                // ... other fields
            };
            
            experiences.Add(experience);
        }
        
        return experiences;
    }
}
```

---

## Future Work: Multi-Branch Model Implementation

The infrastructure is ready, but **full integration requires implementing multi-branch neural network architectures**. Here's what needs to be done:

### Step 1: Design Multi-Branch Architecture

```python
# Conceptual PyTorch/ONNX model architecture
class MultiBranchModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 5-minute branch (strategic features)
        self.branch_5m = nn.Sequential(
            nn.Linear(num_features_5m, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 1-minute branch (tactical features)
        self.branch_1m = nn.Sequential(
            nn.Linear(num_features_1m, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Fusion layer (combines both branches)
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: single prediction
        )
    
    def forward(self, features_5m, features_1m):
        # Process each timeframe independently
        branch_5m_out = self.branch_5m(features_5m)
        branch_1m_out = self.branch_1m(features_1m)
        
        # Concatenate and fuse
        combined = torch.cat([branch_5m_out, branch_1m_out], dim=1)
        output = self.fusion(combined)
        
        return output
```

### Step 2: Modify Existing Trainers

```csharp
// Example: Modify CVaRPPOTrainer to support multi-timeframe

public class CVaRPPOTrainer
{
    // NEW: Multi-timeframe training method
    public async Task TrainWithMultiTimeframeAsync(
        MultiTimeframeTrainingData data,
        CancellationToken cancellationToken)
    {
        foreach (var batch in data.TrainBatches)
        {
            // Pass both feature sets to model
            var predictions = await _model.ForwardAsync(
                batch.Features5m,  // 5-minute features
                batch.Features1m   // 1-minute features
            );
            
            // Compute loss and backpropagate
            var loss = ComputeLoss(predictions, batch.Labels);
            await _model.BackwardAsync(loss);
        }
    }
}
```

### Step 3: Update Model Serialization

```csharp
// Save multi-branch model to ONNX
public class MultiBranchModelSaver
{
    public void SaveToOnnx(string path)
    {
        // Export multi-branch model with two input nodes:
        // - input_5m: [batch_size, num_features_5m]
        // - input_1m: [batch_size, num_features_1m]
        // And one output node:
        // - output: [batch_size, 1]
        
        var inputNames = new[] { "input_5m", "input_1m" };
        var outputNames = new[] { "output" };
        
        torch.onnx.export(
            model,
            (dummy_input_5m, dummy_input_1m),
            path,
            input_names: inputNames,
            output_names: outputNames
        );
    }
}
```

### Step 4: Update Inference Pipeline

```csharp
// Live inference with multi-branch model
public class MultiBranchInferenceEngine
{
    public async Task<double> PredictAsync(
        Dictionary<string, double> features5m,
        Dictionary<string, double> features1m)
    {
        // Convert features to tensors
        var input5m = ConvertToTensor(features5m);
        var input1m = ConvertToTensor(features1m);
        
        // Run inference with both inputs
        var session = new InferenceSession("model.onnx");
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_5m", input5m),
            NamedOnnxValue.CreateFromTensor("input_1m", input1m)
        };
        
        using var results = session.Run(inputs);
        var prediction = results.First().AsTensor<float>().First();
        
        return prediction;
    }
}
```

---

## Integration Checklist

When implementing multi-branch models, ensure you:

- [ ] **Design multi-branch architecture** with separate branches for 5m and 1m features
- [ ] **Update trainers** (CVaRPPO, LSTM, SAC) to accept multi-timeframe batches
- [ ] **Modify model serialization** to save/load multi-input models
- [ ] **Update inference pipeline** to pass both feature sets
- [ ] **Add tests** for multi-timeframe training pipeline
- [ ] **Document model architecture** in model registry
- [ ] **Validate performance** against single-timeframe baseline
- [ ] **Monitor training metrics** for both branches

---

## Testing

### Unit Tests

```csharp
[Fact]
public async Task MultiTimeframePipeline_PrepareData_Success()
{
    // Arrange
    var pipeline = CreatePipeline();
    
    // Act
    var data = await pipeline.PrepareTrainingDataAsync("ES");
    
    // Assert
    Assert.NotEmpty(data.TrainBatches);
    Assert.NotEmpty(data.ValidationBatches);
    Assert.NotEmpty(data.TestBatches);
    Assert.True(data.Statistics.TotalSamples > 0);
}

[Fact]
public void MultiBatch_Features_CorrectShape()
{
    // Arrange
    var batch = CreateSampleBatch();
    
    // Assert
    Assert.Equal(32, batch.BatchSize);  // Batch size
    Assert.Equal(32, batch.Features5m.GetLength(0));  // Rows
    Assert.True(batch.Features5m.GetLength(1) > 0);   // Columns (features)
}
```

### Integration Tests

```csharp
[Fact]
public async Task EndToEnd_DataPreparation_Success()
{
    // Test complete pipeline from loading to batching
    var loader = GetService<MultiTimeframeDataLoader>();
    var assembler = GetService<MultiTimeframeDataAssembler>();
    var batchCreator = GetService<MultiTimeframeBatchCreator>();
    
    // Load data
    var (bars5m, bars1m) = loader.LoadHistoricalData("ES");
    Assert.NotEmpty(bars5m);
    Assert.NotEmpty(bars1m);
    
    // Assemble samples
    var samples = assembler.AssembleSamples("ES", bars5m, bars1m);
    Assert.NotEmpty(samples);
    
    // Create batches
    var batches = batchCreator.CreateBatches(samples);
    Assert.NotEmpty(batches);
}
```

---

## Troubleshooting

### Issue: No samples generated

**Cause**: Insufficient aligned data between 5m and 1m bars  
**Solution**: Verify both JSON files have overlapping time ranges

```bash
# Validate data alignment
python validate-multitimeframe-alignment.py
```

### Issue: Features all zeros

**Cause**: Insufficient lookback period for indicators  
**Solution**: Ensure historical data has at least 30 bars before first sample

### Issue: Labels all same value

**Cause**: Insufficient price movement or wrong threshold  
**Solution**: Adjust threshold in `CalculateLabel` method or verify price data

---

## Performance Considerations

### Memory Usage

- Each `EnhancedMultiTimeframeSample` contains ~100 bars of context
- Batch size of 32 = ~3,200 bars in memory per batch
- Recommended: Keep batch size ≤ 64 for systems with 8GB RAM

### Training Speed

- Batch creation: ~1-2 seconds for 10,000 samples
- Feature extraction: ~5-10 seconds for 90 days of data
- Data loading: ~1-2 seconds per symbol

### Disk Space

- 90 days of 5m bars: ~26,000 bars = ~2MB JSON
- 90 days of 1m bars: ~130,000 bars = ~10MB JSON
- Total per symbol: ~12MB

---

## Summary

✅ **Complete**: All infrastructure components for multi-timeframe training  
✅ **Ready**: Data pipeline from loading to batching  
✅ **Documented**: Full usage examples and integration guide  
⏳ **Future**: Multi-branch model architecture implementation

**Next Steps**:
1. Implement multi-branch neural network architecture
2. Modify existing trainers (CVaRPPO, LSTM, SAC)
3. Update model serialization for multi-input ONNX models
4. Integrate with live inference pipeline
5. Validate performance improvements

---

**Last Updated**: October 2025  
**Component Version**: 1.0.0  
**Feature Hash**: Use `MultiTimeframeFeatureExtractor.GetFeatureVersionHash()`
