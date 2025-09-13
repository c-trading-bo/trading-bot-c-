using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Brains;

/// <summary>
/// Write-only training brain - produces versioned artifacts with no live inference access
/// This ensures complete isolation from live trading operations
/// </summary>
public class TrainingBrain : ITrainingBrain
{
    private readonly ILogger<TrainingBrain> _logger;
    private readonly IModelRegistry _modelRegistry;
    private readonly Dictionary<string, IArtifactBuilder> _artifactBuilders;
    private readonly ConcurrentDictionary<string, TrainingJob> _activeJobs = new();
    private readonly string _stagingPath;

    public TrainingBrain(
        ILogger<TrainingBrain> logger,
        IModelRegistry modelRegistry,
        IEnumerable<IArtifactBuilder> artifactBuilders,
        string? stagingPath = null)
    {
        _logger = logger;
        _modelRegistry = modelRegistry;
        _stagingPath = stagingPath ?? Path.Combine(Directory.GetCurrentDirectory(), "model_staging");
        
        // Index artifact builders by supported model type
        _artifactBuilders = artifactBuilders.ToDictionary(
            builder => builder.SupportedModelType,
            builder => builder,
            StringComparer.OrdinalIgnoreCase);
        
        // Ensure staging directory exists
        Directory.CreateDirectory(_stagingPath);
        
        _logger.LogInformation("TrainingBrain initialized with staging path: {StagingPath}", _stagingPath);
    }

    /// <summary>
    /// Train a new challenger model for an algorithm
    /// This runs in complete isolation from live trading
    /// </summary>
    public async Task<TrainingResult> TrainChallengerAsync(string algorithm, TrainingConfig config, CancellationToken cancellationToken = default)
    {
        var jobId = GenerateJobId(algorithm);
        var job = new TrainingJob
        {
            JobId = jobId,
            Algorithm = algorithm,
            Config = config,
            Status = "QUEUED",
            StartTime = DateTime.UtcNow,
            Progress = 0.0m
        };

        _activeJobs[jobId] = job;
        
        try
        {
            _logger.LogInformation("Starting training job {JobId} for algorithm {Algorithm}", jobId, algorithm);
            
            // Validate training configuration
            await ValidateTrainingConfigAsync(config, cancellationToken);
            
            job.Status = "RUNNING";
            job.CurrentStage = "DATA_PREPARATION";
            
            // Stage 1: Data Preparation (20% progress)
            await PrepareTrainingDataAsync(job, cancellationToken);
            job.Progress = 0.2m;
            
            job.CurrentStage = "MODEL_TRAINING";
            
            // Stage 2: Model Training (60% progress)
            var modelPath = await TrainModelAsync(job, cancellationToken);
            job.Progress = 0.8m;
            
            job.CurrentStage = "ARTIFACT_CREATION";
            
            // Stage 3: Export to artifact (20% progress)
            var metadata = CreateTrainingMetadata(job);
            var modelVersion = await ExportModelAsync(algorithm, modelPath, metadata, cancellationToken);
            job.Progress = 1.0m;
            
            job.Status = "COMPLETED";
            job.EndTime = DateTime.UtcNow;
            job.CurrentStage = "COMPLETE";
            
            var result = new TrainingResult
            {
                JobId = jobId,
                Algorithm = algorithm,
                Success = true,
                ModelPath = modelVersion.ArtifactPath,
                TrainingDuration = job.EndTime.Value - job.StartTime,
                EpochsCompleted = config.MaxEpochs,
                Metadata = metadata,
                Metrics = new Dictionary<string, decimal>
                {
                    ["final_loss"] = 0.01m, // Placeholder
                    ["validation_score"] = 0.95m, // Placeholder
                    ["sharpe_ratio"] = 1.5m // Placeholder
                }
            };
            
            _logger.LogInformation("Training job {JobId} completed successfully in {Duration:F1}s", 
                jobId, result.TrainingDuration.TotalSeconds);
            
            return result;
        }
        catch (Exception ex)
        {
            job.Status = "FAILED";
            job.EndTime = DateTime.UtcNow;
            job.ErrorMessage = ex.Message;
            
            _logger.LogError(ex, "Training job {JobId} failed: {Error}", jobId, ex.Message);
            
            return new TrainingResult
            {
                JobId = jobId,
                Algorithm = algorithm,
                Success = false,
                ErrorMessage = ex.Message,
                TrainingDuration = DateTime.UtcNow - job.StartTime
            };
        }
    }

    /// <summary>
    /// Export a trained model to artifacts and register in model registry
    /// </summary>
    public async Task<ModelVersion> ExportModelAsync(string algorithm, string modelPath, TrainingMetadata metadata, CancellationToken cancellationToken = default)
    {
        try
        {
            // Determine model type and get appropriate artifact builder
            var modelType = DetermineModelType(algorithm, modelPath);
            if (!_artifactBuilders.TryGetValue(modelType, out var artifactBuilder))
            {
                throw new NotSupportedException($"No artifact builder available for model type: {modelType}");
            }

            // Create artifact in staging area
            var versionId = GenerateVersionId(algorithm);
            var artifactFileName = $"{algorithm}_{versionId}.{GetArtifactExtension(modelType)}";
            var artifactPath = Path.Combine(_stagingPath, artifactFileName);
            
            var finalArtifactPath = await artifactBuilder.BuildArtifactAsync(modelPath, artifactPath, metadata, cancellationToken);
            
            // Validate artifact
            if (!await artifactBuilder.ValidateArtifactAsync(finalArtifactPath, cancellationToken))
            {
                throw new InvalidOperationException($"Artifact validation failed for {finalArtifactPath}");
            }

            // Get artifact metadata
            var artifactMetadata = await artifactBuilder.GetArtifactMetadataAsync(finalArtifactPath, cancellationToken);
            
            // Create model version
            var modelVersion = new ModelVersion
            {
                VersionId = versionId,
                Algorithm = algorithm,
                ArtifactPath = finalArtifactPath,
                ArtifactHash = artifactMetadata.Hash,
                GitSha = metadata.GitSha,
                CreatedAt = DateTime.UtcNow,
                CreatedBy = metadata.CreatedBy,
                
                // Training metadata
                TrainingStartTime = metadata.TrainingStartTime,
                TrainingEndTime = metadata.TrainingEndTime,
                DataRangeStart = metadata.DataRangeStart,
                DataRangeEnd = metadata.DataRangeEnd,
                
                // Performance metrics (from training)
                Sharpe = metadata.PerformanceMetrics.GetValueOrDefault("sharpe_ratio", 0),
                Sortino = metadata.PerformanceMetrics.GetValueOrDefault("sortino_ratio", 0),
                CVaR = metadata.PerformanceMetrics.GetValueOrDefault("cvar", 0),
                MaxDrawdown = metadata.PerformanceMetrics.GetValueOrDefault("max_drawdown", 0),
                WinRate = metadata.PerformanceMetrics.GetValueOrDefault("win_rate", 0),
                TotalTrades = (int)metadata.PerformanceMetrics.GetValueOrDefault("total_trades", 0),
                
                // Model schema
                SchemaVersion = "1.0",
                ModelType = modelType,
                Metadata = new Dictionary<string, object>
                {
                    ["artifact_size_bytes"] = artifactMetadata.FileSizeBytes,
                    ["input_shape"] = artifactMetadata.InputShape,
                    ["output_shape"] = artifactMetadata.OutputShape,
                    ["training_samples"] = metadata.DataSamples,
                    ["training_parameters"] = metadata.Parameters
                }
            };

            // Register in model registry
            var registeredVersionId = await _modelRegistry.RegisterModelAsync(modelVersion, cancellationToken);
            modelVersion.VersionId = registeredVersionId;
            
            _logger.LogInformation("Exported model {Algorithm} version {VersionId} to artifact {ArtifactPath}", 
                algorithm, registeredVersionId, finalArtifactPath);
            
            return modelVersion;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to export model {Algorithm} from {ModelPath}", algorithm, modelPath);
            throw;
        }
    }

    /// <summary>
    /// Get training job status
    /// </summary>
    public async Task<TrainingStatus> GetTrainingStatusAsync(string jobId, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        if (!_activeJobs.TryGetValue(jobId, out var job))
        {
            return new TrainingStatus
            {
                JobId = jobId,
                Status = "NOT_FOUND"
            };
        }

        return new TrainingStatus
        {
            JobId = job.JobId,
            Algorithm = job.Algorithm,
            Status = job.Status,
            Progress = job.Progress,
            StartTime = job.StartTime,
            EndTime = job.EndTime,
            CurrentStage = job.CurrentStage,
            Logs = job.Logs.TakeLast(100).ToList(), // Return recent logs
            ErrorMessage = job.ErrorMessage
        };
    }

    /// <summary>
    /// Cancel a training job
    /// </summary>
    public async Task<bool> CancelTrainingAsync(string jobId, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        if (!_activeJobs.TryGetValue(jobId, out var job))
        {
            return false;
        }

        if (job.Status == "RUNNING" || job.Status == "QUEUED")
        {
            job.Status = "CANCELLED";
            job.EndTime = DateTime.UtcNow;
            job.CurrentStage = "CANCELLED";
            
            _logger.LogInformation("Training job {JobId} cancelled", jobId);
            return true;
        }

        return false;
    }

    #region Private Methods

    private async Task ValidateTrainingConfigAsync(TrainingConfig config, CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        
        if (string.IsNullOrEmpty(config.Algorithm))
        {
            throw new ArgumentException("Algorithm is required", nameof(config));
        }

        if (config.DataStartTime >= config.DataEndTime)
        {
            throw new ArgumentException("DataStartTime must be before DataEndTime", nameof(config));
        }

        if (config.MaxEpochs <= 0)
        {
            throw new ArgumentException("MaxEpochs must be positive", nameof(config));
        }

        // Temporal hygiene check - training data must end before current time
        if (config.DataEndTime > DateTime.UtcNow.AddHours(-1))
        {
            throw new ArgumentException("Training data end time must be at least 1 hour in the past to prevent data leakage", nameof(config));
        }
    }

    private async Task PrepareTrainingDataAsync(TrainingJob job, CancellationToken cancellationToken)
    {
        // Simulate data preparation
        await Task.Delay(1000, cancellationToken);
        
        job.Logs.Add($"[{DateTime.UtcNow:HH:mm:ss}] Preparing training data from {job.Config.DataStartTime} to {job.Config.DataEndTime}");
        job.Logs.Add($"[{DateTime.UtcNow:HH:mm:ss}] Data source: {job.Config.DataSource}");
        job.StageData["data_samples"] = 10000; // Placeholder
    }

    private async Task<string> TrainModelAsync(TrainingJob job, CancellationToken cancellationToken)
    {
        // Simulate model training
        var epochs = job.Config.MaxEpochs;
        var modelPath = Path.Combine(_stagingPath, $"{job.Algorithm}_temp_model_{job.JobId}.onnx");
        
        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                throw new OperationCanceledException();
            }

            // Simulate training time
            await Task.Delay(50, cancellationToken);
            
            var epochProgress = (decimal)epoch / epochs;
            job.Progress = 0.2m + (epochProgress * 0.6m); // 20% base + 60% training progress
            
            if (epoch % 10 == 0 || epoch == epochs)
            {
                job.Logs.Add($"[{DateTime.UtcNow:HH:mm:ss}] Epoch {epoch}/{epochs} - Loss: {0.1m / epoch:F4}");
            }
        }

        // Create a placeholder model file
        await File.WriteAllTextAsync(modelPath, "placeholder_onnx_model", cancellationToken);
        
        job.Logs.Add($"[{DateTime.UtcNow:HH:mm:ss}] Training completed - Model saved to {modelPath}");
        return modelPath;
    }

    private TrainingMetadata CreateTrainingMetadata(TrainingJob job)
    {
        return new TrainingMetadata
        {
            TrainingStartTime = job.StartTime,
            TrainingEndTime = job.EndTime ?? DateTime.UtcNow,
            DataRangeStart = job.Config.DataStartTime.ToString("O"),
            DataRangeEnd = job.Config.DataEndTime.ToString("O"),
            DataSamples = (int)job.StageData.GetValueOrDefault("data_samples", 0),
            GitSha = GetCurrentGitSha(),
            CreatedBy = Environment.UserName,
            Parameters = job.Config.Parameters,
            PerformanceMetrics = new Dictionary<string, decimal>
            {
                ["sharpe_ratio"] = 1.5m, // Placeholder
                ["sortino_ratio"] = 1.8m, // Placeholder
                ["max_drawdown"] = -0.05m, // Placeholder
                ["win_rate"] = 0.65m, // Placeholder
                ["total_trades"] = 100 // Placeholder
            }
        };
    }

    private string DetermineModelType(string algorithm, string modelPath)
    {
        return algorithm.ToUpperInvariant() switch
        {
            "PPO" => "ONNX",
            "LSTM" => "ONNX", 
            "UCB" => "UCB",
            _ => Path.GetExtension(modelPath).ToLowerInvariant() switch
            {
                ".onnx" => "ONNX",
                ".json" => "UCB",
                _ => "ONNX" // Default
            }
        };
    }

    private string GetArtifactExtension(string modelType)
    {
        return modelType.ToUpperInvariant() switch
        {
            "ONNX" => "onnx",
            "UCB" => "json",
            _ => "bin"
        };
    }

    private string GenerateJobId(string algorithm)
    {
        return $"{algorithm}_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid().ToString("N")[..8]}";
    }

    private string GenerateVersionId(string algorithm)
    {
        return $"v{DateTime.UtcNow:yyyyMMdd_HHmmss}_{algorithm}_{Guid.NewGuid().ToString("N")[..8]}";
    }

    private string GetCurrentGitSha()
    {
        try
        {
            // In a real implementation, this would get the actual git SHA
            return Environment.GetEnvironmentVariable("GIT_SHA") ?? "unknown";
        }
        catch
        {
            return "unknown";
        }
    }

    #endregion
}

/// <summary>
/// Internal training job tracking
/// </summary>
internal class TrainingJob
{
    public string JobId { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;
    public TrainingConfig Config { get; set; } = new();
    public string Status { get; set; } = "QUEUED";
    public decimal Progress { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public string CurrentStage { get; set; } = string.Empty;
    public List<string> Logs { get; set; } = new();
    public Dictionary<string, object> StageData { get; set; } = new();
    public string? ErrorMessage { get; set; }
}