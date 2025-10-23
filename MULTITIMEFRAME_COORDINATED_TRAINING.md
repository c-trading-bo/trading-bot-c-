# Multi-Timeframe Coordinated Training - Implementation Guide

## Overview

This document describes the new multi-timeframe training components that enable coordinated learning across multiple timeframes (5m + 1m bars).

## New Components

### 1. MultiTimeframeDataAssembler

**Purpose:** Creates synchronized training samples from multiple timeframes.

**What it does:**
- Takes 5-minute and 1-minute historical bars
- Aligns them to common decision points (5m bar close times)
- For each decision point, gathers:
  - Last 36 5-minute bars (3 hours of strategic context)
  - Last 60 1-minute bars (1 hour of tactical context)
- Extracts features from both timeframes
- Creates labels for supervised learning (price direction in next 5 minutes)

**Usage:**

```csharp
// Initialize
var assembler = new MultiTimeframeDataAssembler(logger, featureExtractor);

// Load historical data
var (bars5m, bars1m) = dataLoader.LoadHistoricalData("ES");

// Assemble synchronized samples
var samples = assembler.AssembleSamples("ES", bars5m, bars1m);

// Each sample contains:
// - Context5m: List<BarData> (36 bars)
// - Context1m: List<BarData> (60 bars)
// - Features5m: Dictionary<string, double> (extracted features)
// - Features1m: Dictionary<string, double> (extracted features)
// - Label: double (1.0=up, -1.0=down, 0.0=flat)
```

### 2. MultiTimeframeBatchCreator

**Purpose:** Groups synchronized samples into batches for efficient GPU training.

**What it does:**
- Takes list of synchronized samples
- Groups them into batches (default: 32 samples per batch)
- Converts feature dictionaries to arrays
- Creates attention masks for padding
- Optionally shuffles samples for training

**Usage:**

```csharp
// Initialize
var batchCreator = new MultiTimeframeBatchCreator(logger);

// Create batches from samples
var batches = batchCreator.CreateBatches(
    samples: samples,
    batchSize: 32,
    shuffle: true  // Shuffle for training
);

// Each batch contains:
// - Features5m: double[batch_size, num_features_5m]
// - Features1m: double[batch_size, num_features_1m]
// - Labels: double[batch_size]
// - Masks: int[batch_size, num_features] (for attention)
```

## Integration with Existing Training Pipeline

### Step 1: Assemble Synchronized Samples

In your `HistoricalTrainingOrchestrator` or training service:

```csharp
// After loading historical data
var (bars5m, bars1m) = await LoadHistoricalDataAsync(symbol);

// Assemble synchronized samples
var assembler = serviceProvider.GetRequiredService<MultiTimeframeDataAssembler>();
var allSamples = assembler.AssembleSamples(symbol, bars5m, bars1m);

_logger.LogInformation(
    "[TRAINING] Assembled {Count} synchronized multi-timeframe samples",
    allSamples.Count);
```

### Step 2: Split into Train/Val/Test

Use existing `DynamicDataSplitStrategy` on the synchronized samples:

```csharp
// Calculate total days from sample count
var totalDays = allSamples.Count > 0 
    ? Math.Max(30, allSamples.Count / 288)  // 288 5m bars per day
    : 60;

// Split data (60% train, 20% val, 20% test)
var dataSplit = dataSplitStrategy.SplitData(allSamples, totalDays);

_logger.LogInformation(
    "[TRAINING] Train: {TrainCount}, Val: {ValCount}, Test: {TestCount}",
    dataSplit.TrainData.Count,
    dataSplit.ValidationData.Count,
    dataSplit.TestData.Count);
```

### Step 3: Create Batches

Create batches from each dataset:

```csharp
var batchCreator = serviceProvider.GetRequiredService<MultiTimeframeBatchCreator>();

// Training batches (shuffled)
var trainBatches = batchCreator.CreateBatches(
    dataSplit.TrainData,
    batchSize: 32,
    shuffle: true
);

// Validation batches (not shuffled)
var valBatches = batchCreator.CreateBatches(
    dataSplit.ValidationData,
    batchSize: 32,
    shuffle: false
);

// Test batches (not shuffled)
var testBatches = batchCreator.CreateBatches(
    dataSplit.TestData,
    batchSize: 32,
    shuffle: false
);
```

### Step 4: Train Models with Multi-Branch Architecture

In your trainer service (e.g., `HeavyPhaseTrainerService`):

```csharp
// Training loop
foreach (var batch in trainBatches)
{
    // batch.Features5m: [32, num_features_5m]
    // batch.Features1m: [32, num_features_1m]
    // batch.Labels: [32]
    
    // TODO: Feed to multi-branch model
    // - Branch A processes Features5m (strategic)
    // - Branch B processes Features1m (tactical)
    // - Fusion layer combines both branches
    // - Output head produces predictions
    
    // Forward pass
    var predictions = model.Forward(batch.Features5m, batch.Features1m);
    
    // Calculate loss
    var loss = lossFunction.Calculate(predictions, batch.Labels);
    
    // Backward pass & update
    optimizer.Step(loss);
}
```

## Data Flow Diagram

```
Historical Data (5m + 1m bars)
        ↓
MultiTimeframeDataAssembler
        ↓
Synchronized Samples
(each with 5m context + 1m context + label)
        ↓
DynamicDataSplitStrategy
        ↓
Train/Val/Test Splits
        ↓
MultiTimeframeBatchCreator
        ↓
Batches (ready for GPU)
        ↓
Multi-Branch Model Training
        ↓
Trained Model (ONNX export)
```

## Benefits

### 1. Coordinated Learning
- Models learn cross-timeframe patterns: "5m uptrend + 1m pullback = entry"
- Single model sees both strategic and tactical context
- End-to-end optimization through gradient descent

### 2. Efficient GPU Utilization
- Batched processing (32 samples at once)
- Padded sequences for uniform tensor shapes
- Attention masks handle variable-length sequences

### 3. Clean Integration
- Works with existing `DynamicDataSplitStrategy`
- Compatible with existing overfitting prevention (early stopping, multi-seed)
- Uses existing feature extraction infrastructure

### 4. No Data Leakage
- Synchronized samples respect temporal ordering
- Test set completely isolated
- No lookahead bias in feature extraction

## Example: Complete Sunday Lab Integration

```csharp
public async Task<TrainingSessionResult> RunSundayLabTrainingAsync(
    CancellationToken cancellationToken)
{
    // Step 1: Load historical data
    var (bars5m, bars1m) = await LoadHistoricalDataAsync("ES");
    
    // Step 2: Assemble synchronized samples
    var assembler = _serviceProvider.GetRequiredService<MultiTimeframeDataAssembler>();
    var samples = assembler.AssembleSamples("ES", bars5m, bars1m);
    
    _logger.LogInformation("[LAB] Assembled {Count} multi-timeframe samples", samples.Count);
    
    // Step 3: Split data
    var totalDays = samples.Count / 288;  // 288 5m bars per day
    var dataSplit = _dataSplitStrategy.SplitData(samples, totalDays);
    
    // Step 4: Create batches
    var batchCreator = _serviceProvider.GetRequiredService<MultiTimeframeBatchCreator>();
    var trainBatches = batchCreator.CreateBatches(dataSplit.TrainData, 32, shuffle: true);
    var valBatches = batchCreator.CreateBatches(dataSplit.ValidationData, 32, shuffle: false);
    
    // Step 5: Train models (existing training logic with batches)
    var result = await TrainModelsWithMultiTimeframeBatches(
        trainBatches,
        valBatches,
        cancellationToken);
    
    return result;
}
```

## Next Steps

To fully utilize these components:

1. **Register services in DI container** (`Program.cs`):
   ```csharp
   services.AddSingleton<MultiTimeframeDataAssembler>();
   services.AddSingleton<MultiTimeframeBatchCreator>();
   ```

2. **Modify training loops** to accept batched multi-timeframe data
   
3. **Update model architectures** to multi-branch design (future work)

4. **Add tick data integration** (future enhancement)

## Current Status

✅ **Implemented:**
- MultiTimeframeDataAssembler - Sample assembly
- MultiTimeframeBatchCreator - Batch creation
- EnhancedMultiTimeframeSample - Extended sample class
- MultiTimeframeBatch - Batch data structure

⏳ **Future Work:**
- Multi-branch model architectures
- Tick data integration
- Modified training loops in all trainer services
- Meta-learner approach (alternative to multi-branch)

## Compatibility

These components work seamlessly with:
- ✅ Existing `MultiTimeframeDataLoader`
- ✅ Existing `MultiTimeframeFeatureExtractor`
- ✅ Existing `DynamicDataSplitStrategy`
- ✅ Existing `EarlyStoppingTracker`
- ✅ Existing `MultiSeedTrainingCoordinator`

No breaking changes to existing code - purely additive functionality.
