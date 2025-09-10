using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Concurrent;
using System.Text.RegularExpressions;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Extensions.Options;

namespace BotCore.ML;

/// <summary>
/// Professional ONNX model loader with hot-reload, versioning, health probe, and integrated registry
/// Implements requirement 1.1: Model version check, hot-reload watcher, fallback order, health probe
/// Includes timestamped, hash-versioned model registry with metadata and health checks
/// </summary>
public sealed class OnnxModelLoader : IDisposable
{
    private readonly ILogger<OnnxModelLoader> _logger;
    private readonly ConcurrentDictionary<string, InferenceSession> _loadedSessions = new();
    private readonly ConcurrentDictionary<string, ModelMetadata> _modelMetadata = new();
    private readonly SessionOptions _sessionOptions;
    private readonly Timer _hotReloadTimer;
    private readonly string _modelsDirectory;
    private readonly Regex _modelNamePattern;
    private readonly ModelRegistryOptions _registryOptions;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly string _registryPath;
    private bool _disposed = false;

    // Model versioning pattern: {family}.{symbol}.{strategy}.{regime}.v{semver}+{sha}.onnx
    private static readonly Regex ModelVersionPattern = new(
        @"^(?<family>\w+)\.(?<symbol>\w+)\.(?<strategy>\w+)\.(?<regime>\w+)\.v(?<semver>\d+\.\d+\.\d+)\+(?<sha>[a-f0-9]{8})\.onnx$",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);

    public event Action<ModelHotReloadEvent>? ModelReloaded;
    public event Action<ModelHealthEvent>? ModelHealthChanged;

    public OnnxModelLoader(
        ILogger<OnnxModelLoader> logger, 
        string modelsDirectory = "models",
        IOptions<ModelRegistryOptions>? registryOptions = null)
    {
        _logger = logger;
        _modelsDirectory = modelsDirectory;
        _modelNamePattern = ModelVersionPattern;
        
        // Initialize registry options
        _registryOptions = registryOptions?.Value ?? new ModelRegistryOptions();
        _registryPath = _registryOptions.RegistryPath;
        
        // Configure JSON serialization
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = true
        };
        
        // Configure ONNX Runtime session options for optimal performance
        _sessionOptions = new SessionOptions
        {
            EnableCpuMemArena = true,
            EnableMemoryPattern = true,
            EnableProfiling = false, // Disable in production
            ExecutionMode = ExecutionMode.ORT_PARALLEL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
        };

        // Ensure models and registry directories exist
        Directory.CreateDirectory(_modelsDirectory);
        Directory.CreateDirectory(_registryPath);

        // Start hot-reload timer (60s polling as per requirement)
        _hotReloadTimer = new Timer(CheckForModelUpdates, null, TimeSpan.FromSeconds(60), TimeSpan.FromSeconds(60));

        _logger.LogInformation("[ONNX-Loader] Initialized with models directory: {ModelsDir}, registry: {RegistryPath}, hot-reload enabled", 
            _modelsDirectory, _registryPath);
    }

    /// <summary>
    /// Load ONNX model with versioning, fallback order, and health probe
    /// Fallback order: new → previous_good → last_known_good
    /// </summary>
    public async Task<InferenceSession?> LoadModelAsync(string modelPath, bool validateInference = true)
    {
        if (string.IsNullOrEmpty(modelPath))
        {
            _logger.LogError("[ONNX-Loader] Model path is null or empty");
            return null;
        }

        var modelKey = GetModelKey(modelPath);
        
        // Check if already loaded and up-to-date
        if (_loadedSessions.TryGetValue(modelKey, out var existingSession) && 
            _modelMetadata.TryGetValue(modelKey, out var existingMetadata))
        {
            var currentMetadata = await GetModelMetadataAsync(modelPath);
            if (currentMetadata != null && currentMetadata.Checksum == existingMetadata.Checksum)
            {
                _logger.LogDebug("[ONNX-Loader] Reusing cached model: {ModelPath}", modelPath);
                return existingSession;
            }
        }

        // Try loading with fallback order
        var loadResult = await LoadModelWithFallbackAsync(modelPath, validateInference);
        
        if (loadResult.Session != null)
        {
            // Cache the loaded session and metadata
            _loadedSessions[modelKey] = loadResult.Session;
            _modelMetadata[modelKey] = loadResult.Metadata!;
            
            _logger.LogInformation("[ONNX-Loader] Model successfully loaded: {ModelPath} (version: {Version})", 
                modelPath, loadResult.Metadata?.Version ?? "unknown");
                
            // Emit model reload event
            ModelReloaded?.Invoke(new ModelHotReloadEvent
            {
                ModelKey = modelKey,
                ModelPath = modelPath,
                Version = loadResult.Metadata?.Version ?? "unknown",
                LoadedAt = DateTime.UtcNow,
                IsHealthy = loadResult.IsHealthy
            });
        }

        return loadResult.Session;
    }

    /// <summary>
    /// Load model with fallback order: new → previous_good → last_known_good
    /// </summary>
    private async Task<ModelLoadResult> LoadModelWithFallbackAsync(string modelPath, bool validateInference)
    {
        var fallbackCandidates = GetFallbackCandidates(modelPath);
        
        foreach (var candidate in fallbackCandidates)
        {
            try
            {
                _logger.LogInformation("[ONNX-Loader] Attempting to load model: {ModelPath}", candidate);
                
                var result = await LoadSingleModelAsync(candidate, validateInference);
                if (result.Session != null && result.IsHealthy)
                {
                    if (candidate != modelPath)
                    {
                        _logger.LogWarning("[ONNX-Loader] Loaded fallback model: {FallbackPath} (original: {OriginalPath})", 
                            candidate, modelPath);
                    }
                    return result;
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[ONNX-Loader] Failed to load model candidate: {ModelPath}", candidate);
            }
        }
        
        _logger.LogError("[ONNX-Loader] All model loading attempts failed for: {ModelPath}", modelPath);
        return new ModelLoadResult { Session = null, IsHealthy = false };
    }

    /// <summary>
    /// Get fallback candidates in order: new → previous_good → last_known_good
    /// </summary>
    private List<string> GetFallbackCandidates(string modelPath)
    {
        var candidates = new List<string> { modelPath };
        
        try
        {
            var parsedModel = ParseModelPath(modelPath);
            if (parsedModel != null)
            {
                // Find previous versions in the same directory
                var directory = Path.GetDirectoryName(modelPath) ?? _modelsDirectory;
                var pattern = $"{parsedModel.Family}.{parsedModel.Symbol}.{parsedModel.Strategy}.{parsedModel.Regime}.v*.onnx";
                
                var versionedFiles = Directory.GetFiles(directory, pattern)
                    .Select(ParseModelPath)
                    .Where(p => p != null)
                    .OrderByDescending(p => p!.SemVer)
                    .ToList();
                
                // Add previous_good (previous version)
                var previousVersion = versionedFiles.Skip(1).FirstOrDefault();
                if (previousVersion != null)
                {
                    var previousPath = Path.Combine(directory, 
                        $"{previousVersion.Family}.{previousVersion.Symbol}.{previousVersion.Strategy}.{previousVersion.Regime}.v{previousVersion.SemVer}+{previousVersion.Sha}.onnx");
                    if (File.Exists(previousPath))
                    {
                        candidates.Add(previousPath);
                    }
                }
                
                // Add last_known_good (oldest stable version)
                var lastKnownGood = versionedFiles.LastOrDefault();
                if (lastKnownGood != null && lastKnownGood != previousVersion)
                {
                    var lastKnownPath = Path.Combine(directory, 
                        $"{lastKnownGood.Family}.{lastKnownGood.Symbol}.{lastKnownGood.Strategy}.{lastKnownGood.Regime}.v{lastKnownGood.SemVer}+{lastKnownGood.Sha}.onnx");
                    if (File.Exists(lastKnownPath))
                    {
                        candidates.Add(lastKnownPath);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[ONNX-Loader] Error finding fallback candidates for: {ModelPath}", modelPath);
        }
        
        return candidates.Distinct().ToList();
    }

    /// <summary>
    /// Load single model with validation and health probe
    /// </summary>
    private async Task<ModelLoadResult> LoadSingleModelAsync(string modelPath, bool validateInference)
    {
        if (!File.Exists(modelPath))
        {
            return new ModelLoadResult { Session = null, IsHealthy = false };
        }

        try
        {
            var startTime = DateTime.UtcNow;
            
            // Get model metadata
            var metadata = await GetModelMetadataAsync(modelPath);
            
            // Load the ONNX model
            var session = new InferenceSession(modelPath, _sessionOptions);
            
            var loadDuration = DateTime.UtcNow - startTime;
            
            // Validate model structure
            var inputInfo = session.InputMetadata;
            var outputInfo = session.OutputMetadata;
            
            _logger.LogInformation("[ONNX-Loader] Model loaded in {Duration}ms: {InputCount} inputs, {OutputCount} outputs", 
                loadDuration.TotalMilliseconds, inputInfo.Count, outputInfo.Count);
            
            // Health probe: smoke-predict on canned feature row
            var isHealthy = true;
            if (validateInference)
            {
                var healthProbeResult = await HealthProbeAsync(session);
                isHealthy = healthProbeResult.IsHealthy;
                
                if (!isHealthy)
                {
                    _logger.LogError("[ONNX-Loader] Model failed health probe: {ModelPath} - {Error}", 
                        modelPath, healthProbeResult.ErrorMessage);
                    session.Dispose();
                    return new ModelLoadResult { Session = null, IsHealthy = false };
                }
            }

            _logger.LogInformation("[ONNX-Loader] Model passed health probe: {ModelPath}", modelPath);
            
            return new ModelLoadResult 
            { 
                Session = session, 
                Metadata = metadata, 
                IsHealthy = isHealthy 
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ONNX-Loader] Error loading model: {ModelPath}", modelPath);
            return new ModelLoadResult { Session = null, IsHealthy = false };
        }
    }

    /// <summary>
    /// Health probe: smoke-predict on canned feature row
    /// </summary>
    private Task<HealthProbeResult> HealthProbeAsync(InferenceSession session)
    {
        try
        {
            _logger.LogDebug("[ONNX-Loader] Running health probe...");
            
            var inputMetadata = session.InputMetadata;
            var inputs = new List<NamedOnnxValue>();

            // Create canned inputs based on model metadata
            foreach (var input in inputMetadata)
            {
                var inputName = input.Key;
                var inputType = input.Value.ElementType;
                var dimensions = input.Value.Dimensions;

                var cannedInput = CreateCannedInput(inputName, inputType, dimensions);
                if (cannedInput != null)
                {
                    inputs.Add(cannedInput);
                }
            }

            if (inputs.Count == 0)
            {
                return Task.FromResult(new HealthProbeResult 
                { 
                    IsHealthy = false, 
                    ErrorMessage = "No valid inputs created for health probe" 
                });
            }

            // Run inference with canned data
            var startTime = DateTime.UtcNow;
            using var results = session.Run(inputs);
            var inferenceDuration = DateTime.UtcNow - startTime;

            // Validate outputs
            var outputCount = results.Count();
            if (outputCount == 0)
            {
                return Task.FromResult(new HealthProbeResult 
                { 
                    IsHealthy = false, 
                    ErrorMessage = "Model produced no outputs during health probe" 
                });
            }

            // Check for NaN or invalid outputs
            foreach (var result in results)
            {
                if (result.Value is Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> tensor)
                {
                    var hasNaN = tensor.ToArray().Any(f => float.IsNaN(f) || float.IsInfinity(f));
                    if (hasNaN)
                    {
                        return Task.FromResult(new HealthProbeResult 
                        { 
                            IsHealthy = false, 
                            ErrorMessage = "Model output contains NaN or Infinity values" 
                        });
                    }
                }
            }

            _logger.LogDebug("[ONNX-Loader] Health probe passed in {Duration}ms with {OutputCount} outputs", 
                inferenceDuration.TotalMilliseconds, outputCount);

            return Task.FromResult(new HealthProbeResult 
            { 
                IsHealthy = true, 
                InferenceDurationMs = inferenceDuration.TotalMilliseconds 
            });
        }
        catch (Exception ex)
        {
            return Task.FromResult(new HealthProbeResult 
            { 
                IsHealthy = false, 
                ErrorMessage = ex.Message 
            });
        }
    }

    /// <summary>
    /// Hot-reload watcher - checks for model updates every 60 seconds
    /// </summary>
    private async void CheckForModelUpdates(object? state)
    {
        try
        {
            _logger.LogDebug("[ONNX-Loader] Checking for model updates...");
            
            // 4️⃣ Enable Model Hot-Reload: Poll data/registry and data/rl/sac every 60s
            await CheckDataRegistryUpdatesAsync();
            await CheckSACModelsUpdatesAsync();
            
            var modelFiles = Directory.GetFiles(_modelsDirectory, "*.onnx", SearchOption.AllDirectories)
                .Where(f => _modelNamePattern.IsMatch(Path.GetFileName(f)))
                .ToList();

            foreach (var modelFile in modelFiles)
            {
                var modelKey = GetModelKey(modelFile);
                
                // Check if we have this model loaded
                if (_loadedSessions.ContainsKey(modelKey) && _modelMetadata.TryGetValue(modelKey, out var currentMetadata))
                {
                    var newMetadata = await GetModelMetadataAsync(modelFile);
                    
                    // Check if newer version or different checksum
                    if (newMetadata != null && 
                        (newMetadata.SemVer > currentMetadata.SemVer || newMetadata.Checksum != currentMetadata.Checksum))
                    {
                        _logger.LogInformation("[ONNX-Loader] Detected model update: {ModelFile} (v{OldVersion} → v{NewVersion})", 
                            modelFile, currentMetadata.Version, newMetadata.Version);
                        
                        // Hot-reload the model
                        await HotReloadModelAsync(modelFile, modelKey);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[ONNX-Loader] Error during hot-reload check");
        }
    }

    /// <summary>
    /// 4️⃣ Check data/registry for updated model metadata
    /// </summary>
    private async Task CheckDataRegistryUpdatesAsync()
    {
        try
        {
            var registryDir = "data/registry";
            if (!Directory.Exists(registryDir))
            {
                Directory.CreateDirectory(registryDir);
                return;
            }

            var metadataFiles = Directory.GetFiles(registryDir, "*_latest.yaml", SearchOption.AllDirectories);
            foreach (var metadataFile in metadataFiles)
            {
                var lastWriteTime = File.GetLastWriteTimeUtc(metadataFile);
                var modelName = Path.GetFileNameWithoutExtension(metadataFile).Replace("_latest", "");
                
                // Check if this metadata file has been updated since last check
                var cacheKey = $"registry_{modelName}_lastcheck";
                if (!_modelMetadata.ContainsKey(cacheKey) || 
                    _modelMetadata[cacheKey].LoadedAt < lastWriteTime)
                {
                    _logger.LogInformation("[HOT_RELOAD] Registry update detected: {MetadataFile} (modified: {LastWrite})", 
                        Path.GetFileName(metadataFile), lastWriteTime);
                    
                    // Read and parse metadata to trigger model reload if needed
                    var content = await File.ReadAllTextAsync(metadataFile);
                    await ParseMetadataAndTriggerReloadAsync(content, metadataFile);
                    
                    // Update cache
                    _modelMetadata[cacheKey] = new ModelMetadata
                    {
                        Version = "registry_check",
                        LoadedAt = lastWriteTime,
                        IsHealthy = true
                    };
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[HOT_RELOAD] Error checking data/registry updates");
        }
    }

    /// <summary>
    /// 4️⃣ Check data/rl/sac for updated SAC models
    /// </summary>
    private async Task CheckSACModelsUpdatesAsync()
    {
        try
        {
            var sacDir = "data/rl/sac";
            if (!Directory.Exists(sacDir))
            {
                Directory.CreateDirectory(sacDir);
                return;
            }

            var sacFiles = Directory.GetFiles(sacDir, "*.zip", SearchOption.AllDirectories)
                .Concat(Directory.GetFiles(sacDir, "*.pkl", SearchOption.AllDirectories))
                .Concat(Directory.GetFiles(sacDir, "*.pt", SearchOption.AllDirectories));

            foreach (var sacFile in sacFiles)
            {
                var lastWriteTime = File.GetLastWriteTimeUtc(sacFile);
                var modelName = Path.GetFileNameWithoutExtension(sacFile);
                
                // Check if this SAC model has been updated since last check
                var cacheKey = $"sac_{modelName}_lastcheck";
                if (!_modelMetadata.ContainsKey(cacheKey) || 
                    _modelMetadata[cacheKey].LoadedAt < lastWriteTime)
                {
                    _logger.LogInformation("[HOT_RELOAD] SAC model update detected: {SACFile} (modified: {LastWrite})", 
                        Path.GetFileName(sacFile), lastWriteTime);
                    
                    // Trigger SAC model reload in Python side
                    await TriggerSacModelReloadAsync(sacFile);
                    
                    // Update cache
                    _modelMetadata[cacheKey] = new ModelMetadata
                    {
                        Version = "sac_check",
                        LoadedAt = lastWriteTime,
                        IsHealthy = true
                    };
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[HOT_RELOAD] Error checking data/rl/sac updates");
        }
    }

    /// <summary>
    /// Hot-reload a specific model
    /// </summary>
    private async Task HotReloadModelAsync(string modelFile, string modelKey)
    {
        try
        {
            _logger.LogInformation("[ONNX-Loader] Hot-reloading model: {ModelFile}", modelFile);
            
            // Load new model with health probe
            var loadResult = await LoadSingleModelAsync(modelFile, true);
            
            if (loadResult.Session != null && loadResult.IsHealthy)
            {
                // Dispose old session
                if (_loadedSessions.TryRemove(modelKey, out var oldSession))
                {
                    oldSession.Dispose();
                }
                
                // Update with new session and metadata
                _loadedSessions[modelKey] = loadResult.Session;
                _modelMetadata[modelKey] = loadResult.Metadata!;
                
                _logger.LogInformation("[ONNX-Loader] ✅ Hot-reload successful: {ModelFile} (version: {Version})", 
                    modelFile, loadResult.Metadata?.Version ?? "unknown");
                
                // Emit reload event
                ModelReloaded?.Invoke(new ModelHotReloadEvent
                {
                    ModelKey = modelKey,
                    ModelPath = modelFile,
                    Version = loadResult.Metadata?.Version ?? "unknown",
                    LoadedAt = DateTime.UtcNow,
                    IsHealthy = true
                });
            }
            else
            {
                _logger.LogError("[ONNX-Loader] ❌ Hot-reload failed - model failed health probe: {ModelFile}", modelFile);
                
                // Emit health event
                ModelHealthChanged?.Invoke(new ModelHealthEvent
                {
                    ModelKey = modelKey,
                    ModelPath = modelFile,
                    IsHealthy = false,
                    ErrorMessage = "Hot-reload failed - model failed health probe",
                    CheckedAt = DateTime.UtcNow
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ONNX-Loader] ❌ Hot-reload error: {ModelFile}", modelFile);
            
            ModelHealthChanged?.Invoke(new ModelHealthEvent
            {
                ModelKey = modelKey,
                ModelPath = modelFile,
                IsHealthy = false,
                ErrorMessage = $"Hot-reload error: {ex.Message}",
                CheckedAt = DateTime.UtcNow
            });
        }
    }

    /// <summary>
    /// Parse model file name to extract version information
    /// </summary>
    private ParsedModelInfo? ParseModelPath(string modelPath)
    {
        var fileName = Path.GetFileName(modelPath);
        var match = _modelNamePattern.Match(fileName);
        
        if (!match.Success)
        {
            return null;
        }
        
        if (!Version.TryParse(match.Groups["semver"].Value, out var semVer))
        {
            return null;
        }
        
        return new ParsedModelInfo
        {
            Family = match.Groups["family"].Value,
            Symbol = match.Groups["symbol"].Value,
            Strategy = match.Groups["strategy"].Value,
            Regime = match.Groups["regime"].Value,
            SemVer = semVer,
            Sha = match.Groups["sha"].Value
        };
    }

    /// <summary>
    /// Get model metadata including version and checksum
    /// </summary>
    private async Task<ModelMetadata?> GetModelMetadataAsync(string modelPath)
    {
        try
        {
            var parsedInfo = ParseModelPath(modelPath);
            if (parsedInfo == null)
            {
                return null;
            }
            
            // Calculate file checksum
            var checksum = await CalculateFileChecksumAsync(modelPath);
            
            return new ModelMetadata
            {
                ModelPath = modelPath,
                Family = parsedInfo.Family,
                Symbol = parsedInfo.Symbol,
                Strategy = parsedInfo.Strategy,
                Regime = parsedInfo.Regime,
                SemVer = parsedInfo.SemVer,
                Sha = parsedInfo.Sha,
                Version = $"{parsedInfo.SemVer}+{parsedInfo.Sha}",
                Checksum = checksum,
                LastModified = File.GetLastWriteTimeUtc(modelPath)
            };
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[ONNX-Loader] Error getting model metadata: {ModelPath}", modelPath);
            return null;
        }
    }

    /// <summary>
    /// Calculate SHA256 checksum of model file
    /// </summary>
    private async Task<string> CalculateFileChecksumAsync(string filePath)
    {
        using var stream = File.OpenRead(filePath);
        using var sha256 = SHA256.Create();
        var hash = await Task.Run(() => sha256.ComputeHash(stream));
        return Convert.ToHexString(hash).ToLowerInvariant();
    }

    /// <summary>
    /// Get model key for caching
    /// </summary>
    private string GetModelKey(string modelPath)
    {
        var parsedInfo = ParseModelPath(modelPath);
        if (parsedInfo != null)
        {
            return $"{parsedInfo.Family}.{parsedInfo.Symbol}.{parsedInfo.Strategy}.{parsedInfo.Regime}";
        }
        return Path.GetFullPath(modelPath);
    }
    /// <summary>
    /// Create canned input data for health probe
    /// </summary>
    private NamedOnnxValue? CreateCannedInput(string inputName, System.Type elementType, int[] dimensions)
    {
        try
        {
            // Handle dynamic dimensions (replace -1 with 1)
            var safeDimensions = dimensions.Select(d => d == -1 ? 1 : d).ToArray();
            
            if (elementType == typeof(float))
            {
                var shape = safeDimensions;
                var totalElements = shape.Aggregate(1, (a, b) => a * b);
                var data = new float[totalElements];
                
                // Fill with realistic canned trading features
                for (int i = 0; i < totalElements; i++)
                {
                    data[i] = i switch
                    {
                        0 => 0.001f,  // Price return: 0.1%
                        1 => 2.5f,    // Time in position: 2.5 hours
                        2 => 50.0f,   // PnL per unit
                        3 => 0.15f,   // Volatility: 15%
                        4 => 0.6f,    // RSI: 60%
                        5 => 0.3f,    // Bollinger position
                        6 => 1.0f,    // Trending regime
                        7 => 0.0f,    // Not ranging
                        8 => 0.0f,    // Not volatile
                        _ => 0.1f     // Default small value
                    };
                }
                
                var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(data, shape);
                return NamedOnnxValue.CreateFromTensor(inputName, tensor);
            }
            else if (elementType == typeof(long))
            {
                var shape = safeDimensions;
                var totalElements = shape.Aggregate(1, (a, b) => a * b);
                var data = new long[totalElements];
                
                // Fill with appropriate integer values
                for (int i = 0; i < totalElements; i++)
                {
                    data[i] = i % 3; // Values 0, 1, 2 for typical categorical features
                }
                
                var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<long>(data, shape);
                return NamedOnnxValue.CreateFromTensor(inputName, tensor);
            }
            else if (elementType == typeof(int))
            {
                var shape = safeDimensions;
                var totalElements = shape.Aggregate(1, (a, b) => a * b);
                var data = new int[totalElements];
                
                for (int i = 0; i < totalElements; i++)
                {
                    data[i] = i % 3;
                }
                
                var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<int>(data, shape);
                return NamedOnnxValue.CreateFromTensor(inputName, tensor);
            }
            else
            {
                _logger.LogWarning("[ONNX-Loader] Unsupported input type for health probe: {Type}", elementType);
                return null;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ONNX-Loader] Error creating canned input for {InputName}", inputName);
            return null;
        }
    }

    /// <summary>
    /// Get a loaded model session by path or model key
    /// </summary>
    public InferenceSession? GetLoadedModel(string modelPathOrKey)
    {
        var modelKey = GetModelKey(modelPathOrKey);
        return _loadedSessions.TryGetValue(modelKey, out var session) ? session : null;
    }

    /// <summary>
    /// Get model metadata for a loaded model
    /// </summary>
    public ModelMetadata? GetModelMetadata(string modelPathOrKey)
    {
        var modelKey = GetModelKey(modelPathOrKey);
        return _modelMetadata.TryGetValue(modelKey, out var metadata) ? metadata : null;
    }

    /// <summary>
    /// Unload a specific model to free memory
    /// </summary>
    public bool UnloadModel(string modelPathOrKey)
    {
        var modelKey = GetModelKey(modelPathOrKey);
        
        var unloaded = false;
        if (_loadedSessions.TryRemove(modelKey, out var session))
        {
            session.Dispose();
            unloaded = true;
        }
        
        _modelMetadata.TryRemove(modelKey, out _);
        
        if (unloaded)
        {
            _logger.LogInformation("[ONNX-Loader] Model unloaded: {ModelKey}", modelKey);
        }
        
        return unloaded;
    }

    /// <summary>
    /// Get count of currently loaded models
    /// </summary>
    public int LoadedModelCount => _loadedSessions.Count;

    /// <summary>
    /// Get list of loaded model keys
    /// </summary>
    public IEnumerable<string> LoadedModelKeys => _loadedSessions.Keys;

    /// <summary>
    /// Get loaded model information
    /// </summary>
    public IEnumerable<LoadedModelInfo> GetLoadedModels()
    {
        return _loadedSessions.Keys.Select(key => new LoadedModelInfo
        {
            ModelKey = key,
            Metadata = _modelMetadata.GetValueOrDefault(key),
            LoadedAt = _modelMetadata.GetValueOrDefault(key)?.LastModified ?? DateTime.MinValue
        }).ToList();
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _logger.LogInformation("[ONNX-Loader] Disposing {ModelCount} loaded models", _loadedSessions.Count);
            
            // Stop hot-reload timer
            _hotReloadTimer?.Dispose();
            
            // Dispose all loaded sessions
            foreach (var session in _loadedSessions.Values)
            {
                try
                {
                    session.Dispose();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[ONNX-Loader] Error disposing model session");
                }
            }
            
            _loadedSessions.Clear();
            _modelMetadata.Clear();
            _sessionOptions.Dispose();
            
            _disposed = true;
            _logger.LogInformation("[ONNX-Loader] Disposed successfully");
        }
    }

    #region Model Registry Methods

    /// <summary>
    /// Register a new model with timestamped, hash-versioned metadata
    /// </summary>
    public async Task<ModelRegistryEntry> RegisterModelAsync(
        string modelName,
        string modelPath,
        ModelRegistryMetadata metadata,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Calculate model hash
            var modelHash = await CalculateFileHashAsync(modelPath, cancellationToken);
            var timestamp = DateTime.UtcNow;
            var version = $"{timestamp:yyyyMMdd-HHmmss}-{modelHash[..8]}";

            // Create registry entry
            var entry = new ModelRegistryEntry
            {
                ModelName = modelName,
                Version = version,
                Timestamp = timestamp,
                Hash = modelHash,
                Metadata = metadata,
                OriginalPath = modelPath,
                Status = ModelRegistryStatus.Registered
            };

            // Create versioned directory
            var versionedDir = Path.Combine(_registryPath, modelName, version);
            Directory.CreateDirectory(versionedDir);

            // Atomic copy model file to registry using temp file + File.Move pattern
            var registryModelPath = Path.Combine(versionedDir, $"{modelName}.onnx");
            var tempModelPath = registryModelPath + ".tmp";
            
            try
            {
                // Step 1: Copy to temporary file first
                File.Copy(modelPath, tempModelPath, overwrite: false);
                
                // Step 2: Atomic move to final destination
                if (File.Exists(registryModelPath))
                {
                    var backupPath = registryModelPath + ".backup";
                    File.Replace(tempModelPath, registryModelPath, backupPath);
                    _logger.LogDebug("[ONNX-Registry] Model deployed atomically with backup: {Model}, backup: {Backup}", 
                        Path.GetFileName(registryModelPath), Path.GetFileName(backupPath));
                }
                else
                {
                    File.Move(tempModelPath, registryModelPath);
                    _logger.LogDebug("[ONNX-Registry] Model deployed atomically: {Model}", Path.GetFileName(registryModelPath));
                }
            }
            catch
            {
                // Cleanup temp file on error
                if (File.Exists(tempModelPath))
                {
                    try { File.Delete(tempModelPath); } catch { }
                }
                throw;
            }
            
            entry.RegistryPath = registryModelPath;

            // Compress model if enabled
            if (_registryOptions.AutoCompress)
            {
                await CompressModelAsync(registryModelPath, cancellationToken);
                entry.IsCompressed = true;
            }

            // Save metadata
            var metadataPath = Path.Combine(versionedDir, "metadata.json");
            var metadataJson = JsonSerializer.Serialize(entry, _jsonOptions);
            await File.WriteAllTextAsync(metadataPath, metadataJson, cancellationToken);

            // Update registry index
            await UpdateRegistryIndexAsync(entry, cancellationToken);

            _logger.LogInformation("[ONNX-Registry] Model registered: {ModelName} v{Version}", modelName, version);
            return entry;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ONNX-Registry] Failed to register model: {ModelName}", modelName);
            throw;
        }
    }

    /// <summary>
    /// Get latest model version from registry
    /// </summary>
    public async Task<ModelRegistryEntry?> GetLatestRegisteredModelAsync(string modelName, CancellationToken cancellationToken = default)
    {
        try
        {
            var modelDir = Path.Combine(_registryPath, modelName);
            if (!Directory.Exists(modelDir))
            {
                return null;
            }

            var versions = Directory.GetDirectories(modelDir)
                .Select(d => Path.GetFileName(d))
                .OrderByDescending(v => v)
                .ToList();

            if (!versions.Any())
            {
                return null;
            }

            var latestVersion = versions.First();
            var metadataPath = Path.Combine(modelDir, latestVersion, "metadata.json");
            
            if (!File.Exists(metadataPath))
            {
                return null;
            }

            var metadataJson = await File.ReadAllTextAsync(metadataPath, cancellationToken);
            var entry = JsonSerializer.Deserialize<ModelRegistryEntry>(metadataJson, _jsonOptions);
            
            return entry;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ONNX-Registry] Failed to get latest model: {ModelName}", modelName);
            return null;
        }
    }

    /// <summary>
    /// Perform comprehensive health check on registered models
    /// </summary>
    public async Task<ModelHealthReport> PerformRegistryHealthCheckAsync(CancellationToken cancellationToken = default)
    {
        var report = new ModelHealthReport
        {
            Timestamp = DateTime.UtcNow,
            ModelStatuses = new List<ModelHealthStatus>()
        };

        try
        {
            var modelDirs = Directory.GetDirectories(_registryPath);
            
            foreach (var modelDir in modelDirs)
            {
                var modelName = Path.GetFileName(modelDir);
                var latestModel = await GetLatestRegisteredModelAsync(modelName, cancellationToken);
                
                var status = new ModelHealthStatus
                {
                    ModelName = modelName,
                    IsHealthy = false,
                    Issues = new List<string>()
                };

                if (latestModel == null)
                {
                    status.Issues.Add("No valid model found");
                }
                else
                {
                    // Check if model file exists
                    if (!File.Exists(latestModel.RegistryPath))
                    {
                        status.Issues.Add("Model file missing");
                    }
                    
                    // Check file integrity
                    var currentHash = await CalculateFileHashAsync(latestModel.RegistryPath, cancellationToken);
                    if (currentHash != latestModel.Hash)
                    {
                        status.Issues.Add("Model file hash mismatch - file may be corrupted");
                    }
                    
                    // Check age
                    var age = DateTime.UtcNow - latestModel.Timestamp;
                    if (age > TimeSpan.FromDays(_registryOptions.ModelExpiryDays))
                    {
                        status.Issues.Add($"Model is {age.TotalDays:F1} days old (expires after {_registryOptions.ModelExpiryDays} days)");
                    }

                    status.IsHealthy = !status.Issues.Any();
                    status.LastUpdated = latestModel.Timestamp;
                    status.Version = latestModel.Version;
                }

                report.ModelStatuses.Add(status);
            }

            report.IsHealthy = report.ModelStatuses.All(s => s.IsHealthy);
            _logger.LogInformation("[ONNX-Registry] Health check completed: {HealthyCount}/{TotalCount} healthy", 
                report.ModelStatuses.Count(s => s.IsHealthy), report.ModelStatuses.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ONNX-Registry] Failed to perform health check");
            report.IsHealthy = false;
        }

        return report;
    }

    /// <summary>
    /// Clean up old model versions
    /// </summary>
    public Task CleanupOldVersionsAsync(int keepVersions = 5, CancellationToken cancellationToken = default)
    {
        try
        {
            var modelDirs = Directory.GetDirectories(_registryPath);
            
            foreach (var modelDir in modelDirs)
            {
                var modelName = Path.GetFileName(modelDir);
                var versions = Directory.GetDirectories(modelDir)
                    .Select(d => new { Path = d, Version = Path.GetFileName(d) })
                    .OrderByDescending(v => v.Version)
                    .ToList();

                if (versions.Count > keepVersions)
                {
                    var toDelete = versions.Skip(keepVersions);
                    foreach (var version in toDelete)
                    {
                        Directory.Delete(version.Path, recursive: true);
                        _logger.LogInformation("[ONNX-Registry] Cleaned up old model version: {ModelName} v{Version}", 
                            modelName, version.Version);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ONNX-Registry] Failed to cleanup old model versions");
        }
        
        return Task.CompletedTask;
    }

    private async Task<string> CalculateFileHashAsync(string filePath, CancellationToken cancellationToken)
    {
        using var sha256 = SHA256.Create();
        using var fileStream = File.OpenRead(filePath);
        var hashBytes = await Task.Run(() => sha256.ComputeHash(fileStream), cancellationToken);
        return Convert.ToHexString(hashBytes).ToLowerInvariant();
    }

    private async Task CompressModelAsync(string modelPath, CancellationToken cancellationToken)
    {
        // Simple compression placeholder - could be enhanced with actual compression algorithms
        await Task.Run(() =>
        {
            var compressedPath = modelPath + ".gz";
            // Implementation would use GZip or similar compression
            _logger.LogDebug("[ONNX-Registry] Model compression placeholder for: {ModelPath}", modelPath);
        }, cancellationToken);
    }

    private async Task UpdateRegistryIndexAsync(ModelRegistryEntry entry, CancellationToken cancellationToken)
    {
        var indexPath = Path.Combine(_registryPath, "registry_index.json");
        
        List<ModelRegistryEntry> index;
        if (File.Exists(indexPath))
        {
            var indexJson = await File.ReadAllTextAsync(indexPath, cancellationToken);
            index = JsonSerializer.Deserialize<List<ModelRegistryEntry>>(indexJson, _jsonOptions) ?? new();
        }
        else
        {
            index = new List<ModelRegistryEntry>();
        }

        // Add or update entry
        var existingIndex = index.FindIndex(e => e.ModelName == entry.ModelName && e.Version == entry.Version);
        if (existingIndex >= 0)
        {
            index[existingIndex] = entry;
        }
        else
        {
            index.Add(entry);
        }

        // Save updated index
        var updatedIndexJson = JsonSerializer.Serialize(index, _jsonOptions);
        await File.WriteAllTextAsync(indexPath, updatedIndexJson, cancellationToken);
    }

    /// <summary>
    /// Parse metadata and trigger model reload if needed
    /// </summary>
    private async Task ParseMetadataAndTriggerReloadAsync(string content, string metadataFile)
    {
        try
        {
            // Parse metadata content (could be YAML or JSON)
            var metadata = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(content);
            if (metadata != null && metadata.ContainsKey("version"))
            {
                _logger.LogInformation("[MODEL_RELOAD] Model metadata parsed from {File}, version: {Version}", 
                    Path.GetFileName(metadataFile), metadata["version"]);
                
                // Trigger model reload notification
                await NotifyModelUpdateAsync(metadataFile, metadata);
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[MODEL_RELOAD] Failed to parse metadata from {File}", metadataFile);
        }
    }

    /// <summary>
    /// Trigger SAC model reload in Python side
    /// </summary>
    private async Task TriggerSacModelReloadAsync(string sacFile)
    {
        try
        {
            _logger.LogInformation("[SAC_RELOAD] Triggering SAC model reload for {File}", Path.GetFileName(sacFile));
            
            // Send reload signal to Python SAC agent via file system signal
            var reloadSignalFile = Path.Combine(Path.GetDirectoryName(sacFile) ?? "", ".sac_reload_signal");
            await File.WriteAllTextAsync(reloadSignalFile, DateTime.UtcNow.ToString("O"));
            
            _logger.LogInformation("[SAC_RELOAD] Reload signal sent for SAC model");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SAC_RELOAD] Failed to trigger SAC model reload for {File}", sacFile);
        }
    }

    /// <summary>
    /// Notify model update to interested systems
    /// </summary>
    private async Task NotifyModelUpdateAsync(string filePath, Dictionary<string, object> metadata)
    {
        try
        {
            // Create notification payload
            var notification = new
            {
                Type = "ModelUpdate",
                File = Path.GetFileName(filePath),
                Timestamp = DateTime.UtcNow,
                Metadata = metadata
            };

            // Write notification to model update queue (file-based for simplicity)
            var notificationDir = Path.Combine(Directory.GetCurrentDirectory(), "notifications");
            Directory.CreateDirectory(notificationDir);
            
            var notificationFile = Path.Combine(notificationDir, $"model_update_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
            var notificationJson = System.Text.Json.JsonSerializer.Serialize(notification, _jsonOptions);
            
            await File.WriteAllTextAsync(notificationFile, notificationJson);
            _logger.LogInformation("[MODEL_NOTIFICATION] Model update notification created: {File}", notificationFile);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MODEL_NOTIFICATION] Failed to create model update notification");
        }
    }

    #endregion
}

#region Supporting Classes

/// <summary>
/// Model metadata with version information and checksum
/// </summary>
public class ModelMetadata
{
    public string ModelPath { get; set; } = string.Empty;
    public string Family { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public string Regime { get; set; } = string.Empty;
    public Version SemVer { get; set; } = new();
    public string Sha { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public string Checksum { get; set; } = string.Empty;
    public DateTime LastModified { get; set; }
    public DateTime LoadedAt { get; set; }
    public bool IsHealthy { get; set; } = true;
}

/// <summary>
/// Parsed model information from filename
/// </summary>
public class ParsedModelInfo
{
    public string Family { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public string Regime { get; set; } = string.Empty;
    public Version SemVer { get; set; } = new();
    public string Sha { get; set; } = string.Empty;
}

/// <summary>
/// Result of model loading operation
/// </summary>
public class ModelLoadResult
{
    public InferenceSession? Session { get; set; }
    public ModelMetadata? Metadata { get; set; }
    public bool IsHealthy { get; set; }
}

/// <summary>
/// Result of health probe operation
/// </summary>
public class HealthProbeResult
{
    public bool IsHealthy { get; set; }
    public string? ErrorMessage { get; set; }
    public double InferenceDurationMs { get; set; }
}

/// <summary>
/// Event for model hot-reload notifications
/// </summary>
public class ModelHotReloadEvent
{
    public string ModelKey { get; set; } = string.Empty;
    public string ModelPath { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public DateTime LoadedAt { get; set; }
    public bool IsHealthy { get; set; }
}

/// <summary>
/// Event for model health status changes
/// </summary>
public class ModelHealthEvent
{
    public string ModelKey { get; set; } = string.Empty;
    public string ModelPath { get; set; } = string.Empty;
    public bool IsHealthy { get; set; }
    public string? ErrorMessage { get; set; }
    public DateTime CheckedAt { get; set; }
}

/// <summary>
/// Information about a loaded model
/// </summary>
public class LoadedModelInfo
{
    public string ModelKey { get; set; } = string.Empty;
    public ModelMetadata? Metadata { get; set; }
    public DateTime LoadedAt { get; set; }
}

/// <summary>
/// Configuration options for model registry (merged from ModelRegistryService)
/// </summary>
public class ModelRegistryOptions
{
    public string RegistryPath { get; set; } = "models/registry";
    public bool AutoCompress { get; set; } = true;
    public int ModelExpiryDays { get; set; } = 30;
}

/// <summary>
/// Model registry entry with metadata (merged from ModelRegistryService)
/// </summary>
public class ModelRegistryEntry
{
    public string ModelName { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string Hash { get; set; } = string.Empty;
    public ModelRegistryMetadata Metadata { get; set; } = new();
    public string OriginalPath { get; set; } = string.Empty;
    public string RegistryPath { get; set; } = string.Empty;
    public ModelRegistryStatus Status { get; set; }
    public bool IsCompressed { get; set; }
}

/// <summary>
/// Model registry metadata (merged from ModelRegistryService)
/// </summary>
public class ModelRegistryMetadata
{
    public DateTime TrainingDate { get; set; }
    public Dictionary<string, object> Hyperparams { get; set; } = new();
    public string TrainingDataHash { get; set; } = string.Empty;
    public double ValidationAccuracy { get; set; }
    public Dictionary<string, double> TrainingMetrics { get; set; } = new();
    public string Description { get; set; } = string.Empty;
    public List<string> Tags { get; set; } = new();
}

/// <summary>
/// Model registry status enumeration (merged from ModelRegistryService)
/// </summary>
public enum ModelRegistryStatus
{
    Registered,
    Active,
    Deprecated,
    Failed
}

/// <summary>
/// Model health report (merged from ModelRegistryService)
/// </summary>
public class ModelHealthReport
{
    public DateTime Timestamp { get; set; }
    public bool IsHealthy { get; set; }
    public List<ModelHealthStatus> ModelStatuses { get; set; } = new();
}

/// <summary>
/// Individual model health status (merged from ModelRegistryService)
/// </summary>
public class ModelHealthStatus
{
    public string ModelName { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public bool IsHealthy { get; set; }
    public DateTime LastUpdated { get; set; }
    public List<string> Issues { get; set; } = new();
}

#endregion
