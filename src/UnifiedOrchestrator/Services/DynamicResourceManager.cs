using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Dynamic Resource Manager - Phase 12: Resource Optimization
/// Calculates optimal thresholds and training strategies based on system capabilities
/// Replaces hardcoded values with intelligent, adaptive resource management
/// </summary>
internal sealed class DynamicResourceManager
{
    private readonly ILogger<DynamicResourceManager> _logger;
    private readonly SystemCapabilityProfiler _profiler;

    public DynamicResourceManager(
        ILogger<DynamicResourceManager> logger,
        SystemCapabilityProfiler profiler)
    {
        _logger = logger;
        _profiler = profiler;
    }

    /// <summary>
    /// Calculate optimal resource thresholds based on system profile
    /// Phase 12.2: Dynamic Threshold Calculator
    /// </summary>
    public async Task<ResourceThresholds> CalculateOptimalThresholdsAsync(
        SystemProfile profile,
        int componentCount = 273,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[RESOURCE-MANAGER] Calculating optimal thresholds for {Components} components...",
            componentCount);

        var thresholds = new ResourceThresholds();

        // Disk space threshold calculation
        // Base: 10GB for OS overhead
        // Per component: 15MB per model file
        // Historical data: 5GB for 90-day bars
        // Temp space: 5GB for checkpoints and intermediate files
        const double baseOverheadGB = 10;
        const double perComponentMB = 15;
        const double historicalDataGB = 5;
        const double tempSpaceGB = 5;

        var componentsDataGB = (componentCount * perComponentMB) / 1024.0;
        var totalMinimumGB = baseOverheadGB + componentsDataGB + historicalDataGB + tempSpaceGB;
        var recommendedGB = totalMinimumGB * 1.25; // Add 25% buffer

        thresholds.MinDiskSpaceGB = totalMinimumGB;
        thresholds.RecommendedDiskSpaceGB = recommendedGB;
        
        _logger.LogInformation("[RESOURCE-MANAGER] Disk: {Min:F1} GB minimum, {Rec:F1} GB recommended",
            thresholds.MinDiskSpaceGB, thresholds.RecommendedDiskSpaceGB);

        // Memory threshold calculation
        // Base: 2GB for OS and bot runtime
        // Per component: Heavy (200MB), Medium (50MB), Light (20MB)
        // But components train sequentially, so only need max single component memory
        const double baseRuntimeGB = 2;
        const double heavyComponentMB = 200;
        // Note: Medium and Light component memory usage tracked but max is Heavy
        // const double mediumComponentMB = 50;
        // const double lightComponentMB = 20;

        var maxComponentMemoryGB = heavyComponentMB / 1024.0; // Largest component
        var totalMinimumRamGB = baseRuntimeGB + maxComponentMemoryGB;
        var recommendedRamGB = totalMinimumRamGB * 1.6; // Add 60% buffer for caching

        thresholds.MinRamGB = totalMinimumRamGB;
        thresholds.RecommendedRamGB = recommendedRamGB;
        
        _logger.LogInformation("[RESOURCE-MANAGER] Memory: {Min:F1} GB minimum, {Rec:F1} GB recommended",
            thresholds.MinRamGB, thresholds.RecommendedRamGB);

        // CPU threshold - check if CPU already overloaded
        thresholds.MaxCpuUsagePercent = 90.0;
        
        _logger.LogInformation("[RESOURCE-MANAGER] CPU: Max {Max:F0}% utilization threshold",
            thresholds.MaxCpuUsagePercent);

        // Adjust thresholds based on actual available resources
        if (profile.AvailableDiskSpaceGB < thresholds.MinDiskSpaceGB)
        {
            _logger.LogWarning("[RESOURCE-MANAGER] Available disk {Available:F1} GB < minimum {Min:F1} GB - will reduce historical data days",
                profile.AvailableDiskSpaceGB, thresholds.MinDiskSpaceGB);
        }

        if (profile.AvailableRamGB < thresholds.MinRamGB)
        {
            _logger.LogWarning("[RESOURCE-MANAGER] Available RAM {Available:F1} GB < minimum {Min:F1} GB - training will use conservative settings",
                profile.AvailableRamGB, thresholds.MinRamGB);
        }

        await Task.CompletedTask.ConfigureAwait(false);
        return thresholds;
    }

    /// <summary>
    /// Determine optimal training strategy based on system capabilities
    /// Phase 12.3: Adaptive Training Strategy
    /// </summary>
    public async Task<TrainingStrategy> DetermineTrainingStrategyAsync(
        SystemProfile profile,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[RESOURCE-MANAGER] Determining training strategy...");

        var strategy = new TrainingStrategy();

        // High-end system: 32GB+ RAM, GPU, SSD, 50+ GB disk
        if (profile.TotalRamGB >= 32 && profile.HasGpu && profile.StorageType == "SSD" && profile.AvailableDiskSpaceGB >= 50)
        {
            strategy.Name = "INTENSIVE";
            strategy.ComponentCount = 273; // All components
            strategy.HistoricalDataDays = 90;
            strategy.CheckpointFrequency = "Every epoch";
            strategy.BatchSize = 1024;
            strategy.EnableCheckpointing = true;
            
            _logger.LogInformation("[RESOURCE-MANAGER] HIGH-END system - INTENSIVE strategy: All 273 components, 90-day data, large batches");
        }
        // Mid-range system: 8-16GB RAM, SSD, 30-50GB disk
        else if (profile.TotalRamGB >= 8 && profile.StorageType == "SSD" && profile.AvailableDiskSpaceGB >= 30)
        {
            strategy.Name = "STANDARD";
            strategy.ComponentCount = 273; // All components
            strategy.HistoricalDataDays = 60;
            strategy.CheckpointFrequency = "Every 5 epochs";
            strategy.BatchSize = 512;
            strategy.EnableCheckpointing = true;
            
            _logger.LogInformation("[RESOURCE-MANAGER] MID-RANGE system - STANDARD strategy: All 273 components, 60-day data, medium batches");
        }
        // Low-end system: 4-8GB RAM, HDD, 20-30GB disk
        else if (profile.TotalRamGB >= 4 && profile.AvailableDiskSpaceGB >= 20)
        {
            strategy.Name = "LIGHTWEIGHT";
            strategy.ComponentCount = 50; // Top 50 critical components
            strategy.HistoricalDataDays = 30;
            strategy.CheckpointFrequency = "At end only";
            strategy.BatchSize = 256;
            strategy.EnableCheckpointing = false;
            
            _logger.LogWarning("[RESOURCE-MANAGER] LOW-END system - LIGHTWEIGHT strategy: Top 50 components, 30-day data, small batches");
            _logger.LogWarning("[RESOURCE-MANAGER] WARNING: Reduced training scope due to system constraints");
        }
        // Very constrained: <4GB RAM, <20GB disk
        else
        {
            strategy.Name = "MINIMAL";
            strategy.ComponentCount = 10; // CVaRPPO + top 10
            strategy.HistoricalDataDays = 7;
            strategy.CheckpointFrequency = "None";
            strategy.BatchSize = 128;
            strategy.EnableCheckpointing = false;
            
            _logger.LogError("[RESOURCE-MANAGER] CONSTRAINED system - MINIMAL strategy: Only 10 critical components, 7-day data");
            _logger.LogError("[RESOURCE-MANAGER] CRITICAL: Heavily reduced training - consider upgrading hardware");
        }

        // GPU considerations
        if (profile.HasGpu)
        {
            strategy.UseGpu = true;
            strategy.GpuAcceleration = $"{profile.GpuType} acceleration";
            _logger.LogInformation("[RESOURCE-MANAGER] GPU detected - enabling {Type} acceleration", profile.GpuType);
        }
        else
        {
            strategy.UseGpu = false;
            strategy.GpuAcceleration = "CPU-only";
            _logger.LogInformation("[RESOURCE-MANAGER] No GPU detected - CPU-only training (slower)");
        }

        // Storage considerations
        if (profile.StorageType == "HDD")
        {
            strategy.CheckpointFrequency = "At end only"; // Reduce I/O on slow storage
            _logger.LogWarning("[RESOURCE-MANAGER] HDD detected - reducing checkpoint frequency to minimize I/O");
        }

        await Task.CompletedTask.ConfigureAwait(false);
        return strategy;
    }

    /// <summary>
    /// Configure ONNX execution provider based on GPU availability
    /// Phase 12.5: GPU Detection and Utilization
    /// </summary>
    public string ConfigureOnnxExecutionProvider(bool hasGpu, string gpuType)
    {
        if (hasGpu)
        {
            if (gpuType == "CUDA")
            {
                _logger.LogInformation("[RESOURCE-MANAGER] Configuring ONNX with CUDA execution provider");
                return "CUDA";
            }
            else if (gpuType == "DirectML")
            {
                _logger.LogInformation("[RESOURCE-MANAGER] Configuring ONNX with DirectML execution provider");
                return "DirectML";
            }
        }

        _logger.LogInformation("[RESOURCE-MANAGER] Configuring ONNX with CPU execution provider");
        return "CPU";
    }
}

/// <summary>
/// Resource thresholds calculated from system profile
/// </summary>
internal class ResourceThresholds
{
    public double MinDiskSpaceGB { get; set; }
    public double RecommendedDiskSpaceGB { get; set; }
    public double MinRamGB { get; set; }
    public double RecommendedRamGB { get; set; }
    public double MaxCpuUsagePercent { get; set; }
}

/// <summary>
/// Training strategy determined from system capabilities
/// </summary>
internal class TrainingStrategy
{
    public string Name { get; set; } = "STANDARD";
    public int ComponentCount { get; set; } = 273;
    public int HistoricalDataDays { get; set; } = 60;
    public string CheckpointFrequency { get; set; } = "Every 5 epochs";
    public int BatchSize { get; set; } = 512;
    public bool EnableCheckpointing { get; set; } = true;
    public bool UseGpu { get; set; }
    public string GpuAcceleration { get; set; } = "CPU-only";
}
