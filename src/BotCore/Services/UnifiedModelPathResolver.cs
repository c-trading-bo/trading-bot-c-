using Microsoft.Extensions.Logging;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using TradingBot.Abstractions;

namespace BotCore.Services;

/// <summary>
/// 🔍 UNIFIED MODEL PATH RESOLVER - CROSS-PLATFORM ONNX LOADING 🔍
/// 
/// FIXES THE PROBLEM:
/// - ONNX loader expects Windows paths but gets Linux CI paths
/// - "Index and length" exception when parsing model filenames
/// - Models don't exist, but no graceful fallback to RL brain
/// 
/// SOLUTION:
/// 1. Cross-platform path resolution (Windows/Linux compatible)
/// 2. Robust regex validation for model naming patterns
/// 3. Fallback strategy when models don't exist
/// 4. Environment-aware model root detection
/// 
/// RESULT: Either proper model loading OR graceful fallback to working AI
/// </summary>
public class UnifiedModelPathResolver
{
    private readonly ILogger<UnifiedModelPathResolver> _logger;
    private readonly ModelPathConfig _config;
    
    // Model naming pattern validation
    private static readonly Regex ModelNamePattern = new Regex(
        @"^(?<name>[a-zA-Z0-9_]+)-(?<algorithm>[a-zA-Z0-9_]+)-(?<version>\d+\.\d+\.\d+)\.onnx$",
        RegexOptions.Compiled | RegexOptions.IgnoreCase);
    
    // Platform-specific path handling
    private readonly string _modelRootPath;
    private readonly bool _isWindows;
    
    // Model existence cache
    private readonly Dictionary<string, ModelPathInfo> _pathCache = new();
    private readonly object _cacheLock = new();
    
    public UnifiedModelPathResolver(ILogger<UnifiedModelPathResolver> logger)
    {
        _logger = logger;
        _isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
        
        // Initialize configuration
        _config = new ModelPathConfig
        {
            ModelRootEnvironmentVariable = "MODEL_ROOT",
            DefaultWindowsModelRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "TradingBot", "Models"),
            DefaultLinuxModelRoot = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tradingbot", "models"),
            FallbackEnabled = true,
            ValidationEnabled = true
        };
        
        // Resolve model root path
        _modelRootPath = ResolveModelRootPath();
        
        _logger.LogInformation("🔍 [MODEL-RESOLVER] Initialized - Platform: {Platform}, Root: {Root}", 
            _isWindows ? "Windows" : "Linux", _modelRootPath);
    }
    
    /// <summary>
    /// Resolve model file path with cross-platform compatibility and validation
    /// </summary>
    public ModelResolutionResult ResolveModelPath(string modelIdentifier)
    {
        try
        {
            _logger.LogDebug("🔍 [MODEL-RESOLVER] Resolving path for: {Model}", modelIdentifier);
            
            // Check cache first
            lock (_cacheLock)
            {
                if (_pathCache.TryGetValue(modelIdentifier, out var cachedInfo))
                {
                    if (DateTime.UtcNow - cachedInfo.CachedAt < TimeSpan.FromMinutes(5))
                    {
                        _logger.LogTrace("💾 [MODEL-CACHE] Using cached path for: {Model}", modelIdentifier);
                        return CreateResultFromCachedInfo(cachedInfo);
                    }
                    else
                    {
                        _pathCache.Remove(modelIdentifier);
                    }
                }
            }
            
            // Step 1: Validate model name pattern if validation is enabled
            var validationResult = ValidateModelName(modelIdentifier);
            if (_config.ValidationEnabled && !validationResult.IsValid)
            {
                _logger.LogWarning("⚠️ [MODEL-VALIDATION] Invalid model name pattern: {Model} - {Reason}", 
                    modelIdentifier, validationResult.ValidationMessage);
                
                if (!_config.FallbackEnabled)
                {
                    return new ModelResolutionResult
                    {
                        Success = false,
                        ModelPath = null,
                        ErrorMessage = $"Model name validation failed: {validationResult.ValidationMessage}",
                        FallbackAvailable = false
                    };
                }
            }
            
            // Step 2: Resolve potential paths
            var potentialPaths = GeneratePotentialPaths(modelIdentifier);
            
            // Step 3: Check each path for existence
            foreach (var pathInfo in potentialPaths)
            {
                if (File.Exists(pathInfo.FullPath))
                {
                    _logger.LogInformation("✅ [MODEL-FOUND] Located model: {Model} at {Path}", 
                        modelIdentifier, pathInfo.FullPath);
                    
                    // Cache successful resolution
                    CachePathInfo(modelIdentifier, pathInfo, true);
                    
                    return new ModelResolutionResult
                    {
                        Success = true,
                        ModelPath = pathInfo.FullPath,
                        ModelInfo = ExtractModelInfo(modelIdentifier, pathInfo),
                        FallbackAvailable = true
                    };
                }
                else
                {
                    _logger.LogTrace("🔍 [MODEL-SEARCH] Path not found: {Path}", pathInfo.FullPath);
                }
            }
            
            // Step 4: No model found - check fallback options
            if (_config.FallbackEnabled)
            {
                _logger.LogWarning("⚠️ [MODEL-FALLBACK] Model not found: {Model}, fallback to RL brain available", 
                    modelIdentifier);
                
                // Cache failed resolution
                CachePathInfo(modelIdentifier, new ModelPathInfo 
                { 
                    FullPath = string.Empty,
                    Exists = false,
                    PathType = "fallback"
                }, false);
                
                return new ModelResolutionResult
                {
                    Success = false,
                    ModelPath = null,
                    ErrorMessage = $"Model file not found: {modelIdentifier}",
                    FallbackAvailable = true,
                    FallbackReason = "Model file does not exist, can use RL brain instead"
                };
            }
            
            // No fallback available
            return new ModelResolutionResult
            {
                Success = false,
                ModelPath = null,
                ErrorMessage = $"Model not found and fallback disabled: {modelIdentifier}",
                FallbackAvailable = false
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [MODEL-RESOLVER] Error resolving path for: {Model}", modelIdentifier);
            
            return new ModelResolutionResult
            {
                Success = false,
                ModelPath = null,
                ErrorMessage = $"Path resolution error: {ex.Message}",
                FallbackAvailable = _config.FallbackEnabled
            };
        }
    }
    
    /// <summary>
    /// Get all available models in the model directory
    /// </summary>
    public List<AvailableModel> DiscoverAvailableModels()
    {
        var models = new List<AvailableModel>();
        
        try
        {
            if (!Directory.Exists(_modelRootPath))
            {
                _logger.LogWarning("⚠️ [MODEL-DISCOVERY] Model root directory does not exist: {Path}", _modelRootPath);
                return models;
            }
            
            var onnxFiles = Directory.GetFiles(_modelRootPath, "*.onnx", SearchOption.AllDirectories);
            
            foreach (var filePath in onnxFiles)
            {
                try
                {
                    var fileName = Path.GetFileName(filePath);
                    var validationResult = ValidateModelName(fileName);
                    
                    var model = new AvailableModel
                    {
                        FileName = fileName,
                        FullPath = filePath,
                        RelativePath = Path.GetRelativePath(_modelRootPath, filePath),
                        IsValidName = validationResult.IsValid,
                        ValidationMessage = validationResult.ValidationMessage,
                        FileSize = new FileInfo(filePath).Length,
                        LastModified = File.GetLastWriteTime(filePath)
                    };
                    
                    if (validationResult.IsValid && validationResult.ParsedInfo != null)
                    {
                        model.ModelName = validationResult.ParsedInfo.Name;
                        model.Algorithm = validationResult.ParsedInfo.Algorithm;
                        model.Version = validationResult.ParsedInfo.Version;
                    }
                    
                    models.Add(model);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "⚠️ [MODEL-DISCOVERY] Error processing model file: {Path}", filePath);
                }
            }
            
            _logger.LogInformation("🔍 [MODEL-DISCOVERY] Discovered {Count} models ({Valid} valid)", 
                models.Count, models.Count(m => m.IsValidName));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [MODEL-DISCOVERY] Error discovering models");
        }
        
        return models;
    }
    
    /// <summary>
    /// Create standardized model path for saving new models
    /// </summary>
    public string CreateStandardModelPath(string modelName, string algorithm, string version)
    {
        var fileName = $"{modelName}-{algorithm}-{version}.onnx";
        var fullPath = Path.Combine(_modelRootPath, fileName);
        
        // Ensure directory exists
        Directory.CreateDirectory(_modelRootPath);
        
        _logger.LogDebug("🔍 [MODEL-CREATE] Created standard path: {Path}", fullPath);
        return fullPath;
    }
    
    /// <summary>
    /// Clear path resolution cache
    /// </summary>
    public void ClearCache()
    {
        lock (_cacheLock)
        {
            _pathCache.Clear();
        }
        
        _logger.LogInformation("🧹 [MODEL-CACHE] Path cache cleared");
    }
    
    /// <summary>
    /// Get resolver statistics
    /// </summary>
    public ModelResolverStats GetStats()
    {
        lock (_cacheLock)
        {
            return new ModelResolverStats
            {
                CachedPaths = _pathCache.Count,
                ModelRootPath = _modelRootPath,
                Platform = _isWindows ? "Windows" : "Linux",
                ValidationEnabled = _config.ValidationEnabled,
                FallbackEnabled = _config.FallbackEnabled,
                LastDiscovery = DateTime.UtcNow
            };
        }
    }
    
    #region Private Implementation Methods
    
    private string ResolveModelRootPath()
    {
        // Check environment variable first
        var envPath = Environment.GetEnvironmentVariable(_config.ModelRootEnvironmentVariable);
        if (!string.IsNullOrEmpty(envPath))
        {
            var resolvedEnvPath = ResolvePlatformPath(envPath);
            _logger.LogInformation("🔧 [MODEL-ROOT] Using environment path: {Path}", resolvedEnvPath);
            return resolvedEnvPath;
        }
        
        // Use platform-specific default
        var defaultPath = _isWindows ? _config.DefaultWindowsModelRoot : _config.DefaultLinuxModelRoot;
        var resolvedDefaultPath = ResolvePlatformPath(defaultPath);
        
        _logger.LogInformation("🔧 [MODEL-ROOT] Using default path: {Path}", resolvedDefaultPath);
        return resolvedDefaultPath;
    }
    
    private string ResolvePlatformPath(string path)
    {
        if (string.IsNullOrEmpty(path)) return string.Empty;
        
        // Convert path separators for current platform
        if (_isWindows)
        {
            return path.Replace('/', '\\');
        }
        else
        {
            return path.Replace('\\', '/');
        }
    }
    
    private static ModelNameValidation ValidateModelName(string fileName)
    {
        if (string.IsNullOrEmpty(fileName))
        {
            return new ModelNameValidation
            {
                IsValid = false,
                ValidationMessage = "Model name is null or empty"
            };
        }
        
        try
        {
            var match = ModelNamePattern.Match(fileName);
            if (!match.Success)
            {
                return new ModelNameValidation
                {
                    IsValid = false,
                    ValidationMessage = "Model name does not match expected pattern: name-algorithm-version.onnx"
                };
            }
            
            return new ModelNameValidation
            {
                IsValid = true,
                ValidationMessage = "Valid model name pattern",
                ParsedInfo = new ParsedModelInfo
                {
                    Name = match.Groups["name"].Value,
                    Algorithm = match.Groups["algorithm"].Value,
                    Version = match.Groups["version"].Value
                }
            };
        }
        catch (Exception ex)
        {
            return new ModelNameValidation
            {
                IsValid = false,
                ValidationMessage = $"Error validating model name: {ex.Message}"
            };
        }
    }
    
    private List<ModelPathInfo> GeneratePotentialPaths(string modelIdentifier)
    {
        var paths = new List<ModelPathInfo>();
        
        // Direct path in root
        paths.Add(new ModelPathInfo
        {
            FullPath = Path.Combine(_modelRootPath, modelIdentifier),
            PathType = "direct"
        });
        
        // If not already .onnx, try adding extension
        if (!modelIdentifier.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
        {
            paths.Add(new ModelPathInfo
            {
                FullPath = Path.Combine(_modelRootPath, $"{modelIdentifier}.onnx"),
                PathType = "with_extension"
            });
        }
        
        // Try common subdirectories
        var commonSubdirs = new[] { "onnx", "models", "rl", "ml", "production" };
        foreach (var subdir in commonSubdirs)
        {
            var subdirPath = Path.Combine(_modelRootPath, subdir, modelIdentifier);
            paths.Add(new ModelPathInfo
            {
                FullPath = subdirPath,
                PathType = $"subdir_{subdir}"
            });
            
            if (!modelIdentifier.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase))
            {
                paths.Add(new ModelPathInfo
                {
                    FullPath = $"{subdirPath}.onnx",
                    PathType = $"subdir_{subdir}_with_extension"
                });
            }
        }
        
        // Convert all paths to platform-specific format
        foreach (var path in paths)
        {
            path.FullPath = ResolvePlatformPath(path.FullPath);
        }
        
        return paths;
    }
    
    private void CachePathInfo(string modelIdentifier, ModelPathInfo pathInfo, bool exists)
    {
        lock (_cacheLock)
        {
            pathInfo.Exists = exists;
            pathInfo.CachedAt = DateTime.UtcNow;
            _pathCache[modelIdentifier] = pathInfo;
        }
    }
    
    private ModelResolutionResult CreateResultFromCachedInfo(ModelPathInfo cachedInfo)
    {
        if (cachedInfo.Exists)
        {
            return new ModelResolutionResult
            {
                Success = true,
                ModelPath = cachedInfo.FullPath,
                FallbackAvailable = true
            };
        }
        else
        {
            return new ModelResolutionResult
            {
                Success = false,
                ModelPath = null,
                ErrorMessage = "Model not found (cached result)",
                FallbackAvailable = _config.FallbackEnabled
            };
        }
    }
    
    private TradingBot.Abstractions.ModelInfo ExtractModelInfo(string modelIdentifier, ModelPathInfo pathInfo)
    {
        var validation = ValidateModelName(Path.GetFileName(pathInfo.FullPath));
        
        return new TradingBot.Abstractions.ModelInfo
        {
            ModelIdentifier = modelIdentifier,
            FullPath = pathInfo.FullPath,
            FileName = Path.GetFileName(pathInfo.FullPath),
            IsValidName = validation.IsValid,
            ModelName = validation.ParsedInfo?.Name ?? "unknown",
            Algorithm = validation.ParsedInfo?.Algorithm ?? "unknown",
            Version = validation.ParsedInfo?.Version ?? "unknown",
            FileSize = File.Exists(pathInfo.FullPath) ? new FileInfo(pathInfo.FullPath).Length : 0,
            LastModified = File.Exists(pathInfo.FullPath) ? File.GetLastWriteTime(pathInfo.FullPath) : DateTime.MinValue
        };
    }
    
    #endregion
}

#region Data Models

public class ModelPathConfig
{
    public string ModelRootEnvironmentVariable { get; set; } = "MODEL_ROOT";
    public string DefaultWindowsModelRoot { get; set; } = string.Empty;
    public string DefaultLinuxModelRoot { get; set; } = string.Empty;
    public bool FallbackEnabled { get; set; } = true;
    public bool ValidationEnabled { get; set; } = true;
}

public class ModelResolutionResult
{
    public bool Success { get; set; }
    public string? ModelPath { get; set; }
    public TradingBot.Abstractions.ModelInfo? ModelInfo { get; set; }
    public string? ErrorMessage { get; set; }
    public bool FallbackAvailable { get; set; }
    public string? FallbackReason { get; set; }
}



public class AvailableModel
{
    public string FileName { get; set; } = string.Empty;
    public string FullPath { get; set; } = string.Empty;
    public string RelativePath { get; set; } = string.Empty;
    public bool IsValidName { get; set; }
    public string ValidationMessage { get; set; } = string.Empty;
    public string ModelName { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public long FileSize { get; set; }
    public DateTime LastModified { get; set; }
}

public class ModelNameValidation
{
    public bool IsValid { get; set; }
    public string ValidationMessage { get; set; } = string.Empty;
    public ParsedModelInfo? ParsedInfo { get; set; }
}

public class ParsedModelInfo
{
    public string Name { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
}

public class ModelPathInfo
{
    public string FullPath { get; set; } = string.Empty;
    public string PathType { get; set; } = string.Empty;
    public bool Exists { get; set; }
    public DateTime CachedAt { get; set; }
}

public class ModelResolverStats
{
    public int CachedPaths { get; set; }
    public string ModelRootPath { get; set; } = string.Empty;
    public string Platform { get; set; } = string.Empty;
    public bool ValidationEnabled { get; set; }
    public bool FallbackEnabled { get; set; }
    public DateTime LastDiscovery { get; set; }
}

#endregion