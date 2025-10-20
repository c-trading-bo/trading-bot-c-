# Phase 14: Debugging & Diagnostics Tools - Implementation Guide

## Overview
Phase 14 provides comprehensive debugging and diagnostics tools for Lab Mode training sessions. When enabled, it provides deep visibility into training performance, bottlenecks, and detailed metrics for troubleshooting.

## Implementation Date
2025-10-20

## Components Implemented

### 1. TrainingPerformanceProfiler
**File**: `src/UnifiedOrchestrator/Services/TrainingPerformanceProfiler.cs`

Provides detailed performance profiling with bottleneck identification.

**Features**:
- Section-based timing (start/end profiling sections)
- Automatic bottleneck detection
- Performance report generation
- Call count, average, min, max timing statistics
- Actionable recommendations

**Usage**:
```csharp
// Enable profiling
profiler.StartProfilingSection("DataLoading");
await LoadDataAsync();
profiler.EndProfilingSection("DataLoading");

// Generate report at end of session
var report = await profiler.GenerateProfileReportAsync(sessionId);
```

**Sample Output**:
```
PERFORMANCE PROFILE - Session train-20250119-120000
========================================
Total Time: 5h 23m 49s

Time Breakdown:
- DataLoading                  1h 12m 18s (22.4%)
- ModelTraining                3h 45m 22s (69.7%)
- Validation                      18m 45s ( 5.8%)
- Checkpointing                    7m 24s ( 2.1%)

Detailed Statistics:
Section                        Calls    Avg Time     Min Time     Max Time
--------------------------------------------------------------------------------
DataLoading                      273      15.9s        12.3s        25.1s
ModelTraining                    273      49.4s        38.2s      12m 15s
Validation                       273       4.1s         3.2s         8.7s
Checkpointing                     67       6.6s         5.1s        12.3s

Bottlenecks Identified:
1. Data loading is slow (22.4% of time) - consider caching
2. Model training for Heavy components averages 8.3 minutes (expected 6 minutes)
3. CVaRPPO took 12 minutes (outlier - investigate)

Recommendations:
- Pre-load historical data into memory before training starts
- Optimize data preprocessing pipeline
- Consider GPU acceleration for model training
```

### 2. TrainingDebugLogger
**File**: `src/UnifiedOrchestrator/Services/TrainingDebugLogger.cs`

Provides verbose logging and detailed metrics tracking.

**Features**:
- Verbose before/during/after component logging
- Memory and disk space tracking
- GC collection monitoring
- Data pipeline tracing
- Training metrics JSON export

**Usage**:
```csharp
// Log before component
debugLogger.LogBeforeComponent(
    componentName: "CVaRPPO_Heavy_ES",
    phase: "Heavy",
    componentIndex: 1,
    totalComponents: 273);

// Log during training
debugLogger.LogDuringTraining(
    componentName: "CVaRPPO_Heavy_ES",
    epoch: 10,
    totalEpochs: 100,
    loss: 0.1234,
    learningRate: 0.0001);

// Log after component
debugLogger.LogAfterComponent(
    componentName: "CVaRPPO_Heavy_ES",
    success: true,
    duration: TimeSpan.FromMinutes(8.5),
    metrics: new ComponentDebugMetrics
    {
        FinalLoss = 0.0456,
        BestLoss = 0.0423,
        BestEpoch = 85,
        TotalEpochs = 100,
        ModelSizeMB = 12.5
    });

// Save detailed metrics to file
await debugLogger.LogTrainingMetricsAsync(
    sessionId: "train-123",
    componentName: "CVaRPPO_Heavy_ES",
    metrics: trainingMetrics);
```

**Sample Verbose Output**:
```
[DEBUG] ═══════════════════════════════════════════════════════
[DEBUG] Starting Component: CVaRPPO_Heavy_ES
[DEBUG] Phase: Heavy, Index: 1/273
[DEBUG] Memory: 3.45 GB / 15.89 GB (21.7%)
[DEBUG] Disk Space: 35.2 GB available
[DEBUG] GC Collections: Gen0=125, Gen1=23, Gen2=5

[DEBUG] CVaRPPO_Heavy_ES - Epoch 10/100: Loss=0.123456, LR=0.00010000
[DEBUG] CVaRPPO_Heavy_ES - Epoch 20/100: Loss=0.098765, LR=0.00009500
...

[DEBUG] Completed Component: CVaRPPO_Heavy_ES - ✓ SUCCESS
[DEBUG] Duration: 510.25s
[DEBUG] Final Loss: 0.045600
[DEBUG] Best Loss: 0.042300 (Epoch 85)
[DEBUG] Total Epochs: 100
[DEBUG] Model Size: 12.50 MB
[DEBUG] Memory After: 3.78 GB / 15.89 GB (23.8%)
[DEBUG] ═══════════════════════════════════════════════════════
```

### 3. TrainingMetrics Data Structure
Comprehensive metrics exported to JSON files for detailed analysis.

**File Format**: `artifacts/debug/training-metrics-{sessionId}-{componentName}.json`

**Sample Content**:
```json
{
  "ComponentName": "CVaRPPO_Heavy_ES",
  "SessionId": "train-20250119-120000",
  "StartTime": "2025-01-19T12:05:00Z",
  "EndTime": "2025-01-19T12:13:30Z",
  "DurationSeconds": 510.25,
  
  "TotalEpochs": 100,
  "InitialLoss": 1.2345,
  "FinalLoss": 0.0456,
  "BestLoss": 0.0423,
  "BestEpoch": 85,
  "LossHistory": [1.2345, 1.1234, 1.0123, ...],
  "AverageLoss": 0.3456,
  
  "InitialLearningRate": 0.0001,
  "FinalLearningRate": 0.00005,
  
  "ModelParameterCount": 2450000,
  "ModelSizeMB": 12.5,
  "ModelArchitecture": "PPO_LSTM",
  
  "TrainingSamples": 15000,
  "ValidationSamples": 3000,
  "BatchSize": 512,
  
  "PeakMemoryGB": 4.2,
  "AverageCpuPercent": 78.5,
  "UsedGpu": false,
  "GpuType": "None",
  
  "Converged": true,
  "EpochsToConverge": 85,
  "ConvergenceThreshold": 0.0001,
  
  "SamplesPerSecond": 29.4,
  "TimePerEpochSeconds": 5.1,
  "TimePerBatchMs": 17.3,
  
  "Warnings": [
    "Learning rate reduced at epoch 50",
    "Gradient clipping triggered 3 times"
  ],
  "Errors": []
}
```

## Configuration

### Environment Variables
Enable debugging via environment variables (highest priority):

```bash
# Enable all debugging features
export LAB_DEBUG_MODE=1

# Enable only performance profiling
export LAB_PROFILE=1

# Enable only data pipeline tracing
export LAB_TRACE_DATA=1
```

### appsettings.json
Configure via application settings (lower priority):

```json
{
  "LabDebug": {
    "DebugMode": false,
    "PerformanceProfiling": false,
    "DataPipelineTracing": false,
    "VerboseLogging": false,
    "SaveMetricsToFile": true,
    "DebugOutputDirectory": "artifacts/debug"
  }
}
```

### Priority
Environment variables override appsettings.json configuration.

## Output Files

### Performance Profile
- **Path**: `artifacts/debug/performance-profile.txt`
- **Generated**: At end of training session (if profiling enabled)
- **Contains**: Time breakdown, bottlenecks, recommendations

### Training Metrics
- **Path**: `artifacts/debug/training-metrics-{sessionId}-{componentName}.json`
- **Generated**: After each component training (if debug mode enabled)
- **Contains**: Complete training metrics in JSON format

## Integration Points

### In Training Orchestrator
```csharp
public class HistoricalTrainingOrchestrator
{
    private readonly TrainingPerformanceProfiler _profiler;
    private readonly TrainingDebugLogger _debugLogger;

    public async Task RunTrainingSessionAsync(CancellationToken cancellationToken)
    {
        // Start session profiling
        _profiler.StartProfilingSection("SessionTotal");
        
        // Profile data loading
        _profiler.StartProfilingSection("DataLoading");
        var data = await LoadDataAsync();
        _profiler.EndProfilingSection("DataLoading");
        
        // Train each component
        foreach (var component in components)
        {
            // Debug logging before
            _debugLogger.LogBeforeComponent(
                component.Name, 
                phase, 
                index, 
                total);
            
            // Profile component training
            _profiler.StartProfilingSection($"Train_{component.Name}");
            
            var result = await TrainComponentAsync(component);
            
            _profiler.EndProfilingSection($"Train_{component.Name}");
            
            // Debug logging after
            _debugLogger.LogAfterComponent(
                component.Name, 
                result.Success, 
                result.Duration,
                result.Metrics);
            
            // Save detailed metrics
            if (_debugLogger.IsDebugEnabled)
            {
                await _debugLogger.LogTrainingMetricsAsync(
                    sessionId, 
                    component.Name, 
                    result.DetailedMetrics);
            }
        }
        
        // Generate final profile report
        _profiler.EndProfilingSection("SessionTotal");
        var report = await _profiler.GenerateProfileReportAsync(sessionId);
        _logger.LogInformation("[PROFILER]\n{Report}", report);
    }
}
```

## Use Cases

### 1. Troubleshooting Slow Training
**Problem**: Training takes too long

**Solution**:
```bash
LAB_PROFILE=1 ./run-lab-mode.sh
```

Review `artifacts/debug/performance-profile.txt` to identify:
- Which phase takes most time (DataLoading, ModelTraining, etc.)
- Which components are slowest
- Bottlenecks and recommendations

### 2. Debugging Component Failures
**Problem**: Specific component failing

**Solution**:
```bash
LAB_DEBUG_MODE=1 ./run-lab-mode.sh
```

Review console logs for:
- Memory state before/after component
- Detailed error messages
- GC collection patterns
- Resource constraints

Review `artifacts/debug/training-metrics-{sessionId}-{component}.json` for:
- Loss convergence patterns
- Learning rate schedule
- Model architecture details
- Resource usage peaks

### 3. Optimizing Data Pipeline
**Problem**: Data loading is slow

**Solution**:
```bash
LAB_TRACE_DATA=1 ./run-lab-mode.sh
```

Review data trace logs for:
- Data loading patterns
- Preprocessing bottlenecks
- Cache hit/miss rates
- I/O operations

### 4. Memory Leak Detection
**Problem**: Memory usage growing over time

**Solution**:
```bash
LAB_DEBUG_MODE=1 ./run-lab-mode.sh
```

Monitor console logs for:
- Memory state before/after each component
- GC collection frequency
- Memory growth patterns

## Performance Impact

### Profiling Only (LAB_PROFILE=1)
- **Overhead**: <1% (minimal timer overhead)
- **Disk I/O**: Single report file at end (~50KB)
- **Recommended**: Safe to enable in production

### Debug Mode (LAB_DEBUG_MODE=1)
- **Overhead**: 5-10% (verbose logging, metrics collection)
- **Disk I/O**: Metrics file per component (~273 files * 10KB = 2.7MB)
- **Recommended**: Use for troubleshooting only, not production

### Data Tracing (LAB_TRACE_DATA=1)
- **Overhead**: 2-5% (data pipeline logging)
- **Disk I/O**: Minimal (only console logs)
- **Recommended**: Use when debugging data issues

## Best Practices

### 1. Start with Profiling
Always enable profiling first to identify bottlenecks:
```bash
LAB_PROFILE=1 ./run-lab-mode.sh
```

### 2. Use Debug Mode Selectively
Only enable full debug mode when needed:
```bash
# For specific investigation
LAB_DEBUG_MODE=1 ./run-lab-mode.sh

# Not for routine training
```

### 3. Clean Up Debug Files
Debug files accumulate quickly:
```bash
# Clean old debug files
rm -rf artifacts/debug/*
```

### 4. Review Reports After Training
Performance reports provide actionable insights:
```bash
# View performance report
cat artifacts/debug/performance-profile.txt

# Analyze specific component metrics
jq . artifacts/debug/training-metrics-*-CVaRPPO*.json
```

## Troubleshooting

### Debug Mode Not Activating
**Check**:
1. Environment variable set correctly: `echo $LAB_DEBUG_MODE`
2. Logs show activation: `grep "Debug logging ENABLED" logs/*`

### Performance Report Empty
**Check**:
1. Profiling sections actually called
2. Session completed (report generated at end)
3. File permissions for `artifacts/debug/`

### Metrics Files Not Generated
**Check**:
1. Debug mode enabled
2. Components actually trained
3. Directory writable: `test -w artifacts/debug && echo "OK"`

## Future Enhancements

### Potential Additions
- [ ] Real-time performance dashboard (web UI)
- [ ] Comparison between training sessions
- [ ] Automatic anomaly detection in metrics
- [ ] Integration with external monitoring (Grafana, etc.)
- [ ] Memory profiler integration (dotMemory)
- [ ] Flame graph generation for CPU profiling

## Summary

Phase 14 provides comprehensive debugging and diagnostics capabilities:

✅ **Performance Profiling**: Identify bottlenecks and optimization opportunities
✅ **Verbose Logging**: Deep visibility into training process
✅ **Metrics Export**: Detailed JSON files for analysis
✅ **Minimal Overhead**: <1% with profiling only, 5-10% in full debug mode
✅ **Actionable Reports**: Clear recommendations for improvements
✅ **Flexible Configuration**: Environment variables or appsettings.json

Enable when needed for troubleshooting, disable in production for best performance.
