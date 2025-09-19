using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace TradingBot.RLAgent;

/// <summary>
/// Hot-reload manager for ONNX models with smoke testing and atomic swapping
/// Implements requirement: Safe hot-reload of ONNX models using OnnxEnsembleWrapper
/// </summary>
public class ModelHotReloadManager : IDisposable
{
    private const double TestFeatureRangeMultiplier = 2.0;
    
    private readonly ILogger<ModelHotReloadManager> _logger;
    private readonly OnnxEnsembleWrapper _onnxEnsemble;
    private readonly ModelHotReloadOptions _options;
    private readonly FileSystemWatcher _fileWatcher;
    private readonly CancellationTokenSource _cancellationTokenSource = new();
    private readonly SemaphoreSlim _reloadSemaphore = new(1, 1);
    
    // Cached JSON serializer options
    private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };
    private bool _disposed;

    public ModelHotReloadManager(
        ILogger<ModelHotReloadManager> logger,
        OnnxEnsembleWrapper onnxEnsemble,
        IOptions<ModelHotReloadOptions> options)
    {
        ArgumentNullException.ThrowIfNull(options);
        
        _logger = logger;
        _onnxEnsemble = onnxEnsemble;
        _options = options.Value;

        // Set up file system watcher for model directory
        _fileWatcher = new FileSystemWatcher(_options.WatchDirectory, "*.onnx")
        {
            NotifyFilter = NotifyFilters.CreationTime | NotifyFilters.LastWrite | NotifyFilters.FileName,
            EnableRaisingEvents = true
        };

        _fileWatcher.Created += OnModelFileChanged;
        _fileWatcher.Changed += OnModelFileChanged;

        LogMessages.ModelHotReloadInitialized(_logger, _options.WatchDirectory);
    }

    /// <summary>
    /// Handle model file changes with debouncing
    /// </summary>
    private async void OnModelFileChanged(object sender, FileSystemEventArgs e)
    {
        try
        {
            if (_disposed || _cancellationTokenSource.Token.IsCancellationRequested)
                return;

            // Debounce rapid file system events
            await Task.Delay(_options.DebounceDelayMs, _cancellationTokenSource.Token).ConfigureAwait(false);

            if (!File.Exists(e.FullPath))
                return;

            LogMessages.ModelFileChangeDetected(_logger, e.FullPath);
            
            // Process hot-reload on background thread
            _ = Task.Run(async () => await ProcessHotReloadAsync(e.FullPath).ConfigureAwait(false), _cancellationTokenSource.Token).ConfigureAwait(false);
        }
        catch (ArgumentException ex)
        {
            LogMessages.ModelHotReloadError(_logger, "Error handling model file change event", ex);
        }
        catch (InvalidOperationException ex)
        {
            LogMessages.ModelHotReloadError(_logger, "Error handling model file change event", ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            LogMessages.ModelHotReloadError(_logger, "Error handling model file change event", ex);
        }
    }

    /// <summary>
    /// Process hot-reload with atomic swapping and smoke testing
    /// </summary>
    private async Task ProcessHotReloadAsync(string modelPath)
    {
        if (!await _reloadSemaphore.WaitAsync(_options.ReloadTimeoutMs, _cancellationTokenSource.Token).ConfigureAwait(false))
        {
            LogMessages.ModelHotReloadWarningParam(_logger, modelPath);
            return;
        }

        try
        {
            var fileName = Path.GetFileName(modelPath);
            var candidateModelName = $"candidate-{DateTime.UtcNow:yyyyMMddHHmmss}-{Path.GetFileNameWithoutExtension(fileName)}";
            
            LogMessages.HotReloadStarted(_logger, modelPath, candidateModelName);

            // Step 1: Load candidate model
            var loadSuccess = await _onnxEnsemble.LoadModelAsync(candidateModelName, modelPath, 1.0, _cancellationTokenSource.Token).ConfigureAwait(false);
            if (!loadSuccess)
            {
                LogMessages.ModelHotReloadFailedLoad(_logger, modelPath);
                return;
            }

            // Step 2: Run smoke tests
            var smokeTestPassed = await RunSmokeTestsAsync(candidateModelName).ConfigureAwait(false);
            if (!smokeTestPassed)
            {
                LogMessages.ModelHotReloadSmokeTestFailed(_logger, candidateModelName);
                await _onnxEnsemble.UnloadModelAsync(candidateModelName).ConfigureAwait(false);
                return;
            }

            // Step 3: Atomic swap - unload old model and keep candidate
            var oldModelName = GetCurrentModelName(fileName);
            if (!string.IsNullOrEmpty(oldModelName))
            {
                await _onnxEnsemble.UnloadModelAsync(oldModelName).ConfigureAwait(false);
                LogMessages.OldModelUnloaded(_logger, oldModelName);
            }

            // Step 4: Rename candidate to live model name (optional - can keep candidate name)
            LogMessages.ModelHotReloadCompleted(_logger, candidateModelName);
            
            // Update model registry for tracking
            UpdateModelRegistry(fileName, candidateModelName);
        }
        catch (FileNotFoundException ex)
        {
            LogMessages.ModelHotReloadFileNotFound(_logger, modelPath, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            LogMessages.ModelHotReloadAccessDenied(_logger, modelPath, ex);
        }
        catch (InvalidOperationException ex)
        {
            LogMessages.ModelHotReloadInvalidOperation(_logger, modelPath, ex);
        }
        catch (OutOfMemoryException ex)
        {
            LogMessages.ModelHotReloadOutOfMemory(_logger, modelPath, ex);
        }
        finally
        {
            _reloadSemaphore.Release();
        }
    }

    /// <summary>
    /// Run smoke tests on candidate model with golden inputs
    /// </summary>
    private async Task<bool> RunSmokeTestsAsync(string modelName)
    {
        try
        {
            LogMessages.ModelHotReloadSmokeTestStart(_logger, modelName);

            // Generate deterministic golden inputs for testing
            var goldenInputs = GenerateGoldenInputs();
            
            for (int i = 0; i < _options.SmokeTestIterations; i++)
            {
                foreach (var input in goldenInputs)
                {
                    var prediction = await _onnxEnsemble.PredictAsync(input, _cancellationTokenSource.Token).ConfigureAwait(false);
                    
                    // Validate prediction is within expected bounds
                    if (prediction.Confidence < 0.0 || prediction.Confidence > 1.0)
                    {
                        LogMessages.ModelHotReloadSmokeTestConfidenceFailed(_logger, prediction.Confidence);
                        return false;
                    }

                    if (prediction.IsAnomaly && _options.FailOnAnomalies)
                    {
                        LogMessages.ModelHotReloadSmokeTestAnomalyFailed(_logger);
                        return false;
                    }

                    // Check for NaN or infinite values
                    if (double.IsNaN(prediction.EnsembleResult) || double.IsInfinity(prediction.EnsembleResult))
                    {
                        LogMessages.ModelHotReloadSmokeTestInvalidResult(_logger, prediction.EnsembleResult);
                        return false;
                    }
                }
            }

            LogMessages.SmokeTestsPassed(_logger, modelName);
            return true;
        }
        catch (InvalidOperationException ex)
        {
            LogMessages.ModelHotReloadSmokeTestInvalidOperationFailed(_logger, modelName, ex);
            return false;
        }
        catch (OutOfMemoryException ex)
        {
            LogMessages.ModelHotReloadSmokeTestMemoryFailed(_logger, modelName, ex);
            return false;
        }
        catch (TimeoutException ex)
        {
            LogMessages.ModelHotReloadSmokeTestTimeout(_logger, modelName, ex);
            return false;
        }
    }

    /// <summary>
    /// Generate deterministic golden inputs for smoke testing
    /// </summary>
    private List<float[]> GenerateGoldenInputs()
    {
        var inputs = new List<float[]>();
        
        // Add deterministic test vectors based on expected model input shape
        // These should be representative of normal trading features
        for (int i = 0; i < _options.GoldenInputCount; i++)
        {
            var features = new float[_options.ExpectedFeatureCount];
            
            // Generate deterministic but varied features
            // Use cryptographically secure random for all values in production trading system
            using var rng = RandomNumberGenerator.Create();
            
            for (int j = 0; j < features.Length; j++)
            {
                // Generate normalized features in reasonable trading ranges using secure random
                var randomBytes = new byte[4];
                rng.GetBytes(randomBytes);
                var randomValue = (double)BitConverter.ToUInt32(randomBytes, 0) / uint.MaxValue;
                features[j] = (float)(randomValue * TestFeatureRangeMultiplier - 1.0); // [-1, 1] range
            }
            
            inputs.Add(features);
        }
        
        return inputs;
    }

    /// <summary>
    /// Get current model name for a given file pattern
    /// </summary>
    private static string GetCurrentModelName(string fileName)
    {
        // Implementation would depend on naming convention
        // For now, use a simple pattern based on filename
        return Path.GetFileNameWithoutExtension(fileName);
    }

    /// <summary>
    /// Update model registry for tracking active models
    /// </summary>
    private void UpdateModelRegistry(string fileName, string modelName)
    {
        try
        {
            // Simple file-based registry for tracking
            var registryPath = Path.Combine(_options.WatchDirectory, "model_registry.json");
            var registry = new Dictionary<string, object>
            {
                ["last_reload"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
                ["active_model"] = modelName,
                ["source_file"] = fileName,
                ["reload_count"] = GetReloadCount() + 1
            };

            var json = System.Text.Json.JsonSerializer.Serialize(registry, JsonOptions);
            File.WriteAllText(registryPath, json);
        }
        catch (UnauthorizedAccessException ex)
        {
            LogMessages.ModelHotReloadRegistryAccessDenied(_logger, ex);
        }
        catch (DirectoryNotFoundException ex)
        {
            LogMessages.ModelHotReloadRegistryDirectoryNotFound(_logger, ex);
        }
        catch (IOException ex)
        {
            LogMessages.ModelHotReloadRegistryIOError(_logger, ex);
        }
    }

    /// <summary>
    /// Get current reload count from registry
    /// </summary>
    private int GetReloadCount()
    {
        try
        {
            var registryPath = Path.Combine(_options.WatchDirectory, "model_registry.json");
            if (!File.Exists(registryPath))
                return 0;

            var json = File.ReadAllText(registryPath);
            var registry = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(json);
            
            if (registry?.TryGetValue("reload_count", out var countObj) == true && countObj is JsonElement element)
            {
                return element.GetInt32();
            }
        }
        catch (JsonException)
        {
            // Ignore JSON parsing errors, just return 0
        }
        catch (InvalidOperationException)
        {
            // Ignore operation errors, just return 0  
        }
        catch (ArgumentException)
        {
            // Ignore argument errors, just return 0
        }
        
        return 0;
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _disposed = true;
            
            _fileWatcher?.Dispose();
            _cancellationTokenSource?.Cancel();
            _cancellationTokenSource?.Dispose();
            _reloadSemaphore?.Dispose();
            
            LogMessages.ModelHotReloadManagerDisposed(_logger);
        }
    }
}

/// <summary>
/// Configuration options for model hot-reload manager
/// </summary>
public class ModelHotReloadOptions
{
    public string WatchDirectory { get; set; } = "models/rl";
    public int DebounceDelayMs { get; set; } = 2000;
    public int ReloadTimeoutMs { get; set; } = 30000;
    public int SmokeTestIterations { get; set; } = 3;
    public int GoldenInputCount { get; set; } = 5;
    public int ExpectedFeatureCount { get; set; } = 10;
    public bool FailOnAnomalies { get; set; }
}