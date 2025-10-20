using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Phase 6.3: Baseline Model Manager
/// Stores previous week's production models as baseline for comparison
/// Maintains rolling 4-week history for trend analysis and rollback capability
/// </summary>
internal sealed class BaselineModelManager
{
    private readonly ILogger<BaselineModelManager> _logger;
    private readonly string _baselineDirectory;
    private readonly string _productionDirectory;
    private const int MaxBaselineWeeks = 4;
    
    public BaselineModelManager(ILogger<BaselineModelManager> logger)
    {
        _logger = logger;
        var baseDir = Directory.GetCurrentDirectory();
        _baselineDirectory = Path.Combine(baseDir, "models", "baseline");
        _productionDirectory = Path.Combine(baseDir, "models", "production");
        
        Directory.CreateDirectory(_baselineDirectory);
        Directory.CreateDirectory(_productionDirectory);
    }
    
    /// <summary>
    /// Capture current production models as baseline after successful promotion
    /// Creates snapshot with timestamp for historical tracking
    /// </summary>
    public async Task<string> CaptureBaselineAsync(
        Dictionary<string, decimal> performanceMetrics,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd");
            var baselinePath = Path.Combine(_baselineDirectory, timestamp);
            
            _logger.LogInformation("[BASELINE-MGR] Capturing baseline from production models to {Path}", baselinePath);
            
            // Create baseline directory
            Directory.CreateDirectory(baselinePath);
            
            // Copy all production models to baseline
            var copiedCount = 0;
            if (Directory.Exists(_productionDirectory))
            {
                var productionFiles = Directory.GetFiles(_productionDirectory, "*.onnx", SearchOption.TopDirectoryOnly);
                
                foreach (var sourceFile in productionFiles)
                {
                    var fileName = Path.GetFileName(sourceFile);
                    var destFile = Path.Combine(baselinePath, fileName);
                    File.Copy(sourceFile, destFile, overwrite: true);
                    copiedCount++;
                }
            }
            
            // Save metadata
            var metadata = new BaselineMetadata
            {
                CaptureDate = DateTime.UtcNow,
                ModelCount = copiedCount,
                PerformanceMetrics = performanceMetrics,
                PromotionTimestamp = DateTime.UtcNow,
                Version = "1.0"
            };
            
            var metadataPath = Path.Combine(baselinePath, "metadata.json");
            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(metadata, options);
            await File.WriteAllTextAsync(metadataPath, json, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("[BASELINE-MGR] Captured {Count} models as baseline {Date}", 
                copiedCount, timestamp);
            
            // Cleanup old baselines
            await CleanupOldBaselinesAsync(cancellationToken).ConfigureAwait(false);
            
            return baselinePath;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[BASELINE-MGR] Failed to capture baseline");
            throw;
        }
    }
    
    /// <summary>
    /// Load baseline models from specific date
    /// Returns list of model file paths
    /// </summary>
    public async Task<List<string>> LoadBaselineModelsAsync(
        string date,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var baselinePath = Path.Combine(_baselineDirectory, date);
            
            if (!Directory.Exists(baselinePath))
            {
                _logger.LogWarning("[BASELINE-MGR] Baseline directory not found: {Path}", baselinePath);
                return new List<string>();
            }
            
            var modelFiles = Directory.GetFiles(baselinePath, "*.onnx", SearchOption.TopDirectoryOnly).ToList();
            
            _logger.LogInformation("[BASELINE-MGR] Loaded {Count} baseline models from {Date}", 
                modelFiles.Count, date);
            
            await Task.CompletedTask.ConfigureAwait(false);
            return modelFiles;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[BASELINE-MGR] Failed to load baseline models for date {Date}", date);
            return new List<string>();
        }
    }
    
    /// <summary>
    /// Get the most recent baseline date
    /// Returns null if no baselines exist
    /// </summary>
    public async Task<string?> GetLatestBaselineAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            if (!Directory.Exists(_baselineDirectory))
            {
                return null;
            }
            
            var baselineDirs = Directory.GetDirectories(_baselineDirectory)
                .Select(Path.GetFileName)
                .Where(d => !string.IsNullOrEmpty(d))
                .OrderByDescending(d => d)
                .ToList();
            
            var latest = baselineDirs.FirstOrDefault();
            
            if (latest != null)
            {
                _logger.LogInformation("[BASELINE-MGR] Latest baseline: {Date}", latest);
            }
            
            await Task.CompletedTask.ConfigureAwait(false);
            return latest;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[BASELINE-MGR] Failed to get latest baseline");
            return null;
        }
    }
    
    /// <summary>
    /// Get all available baseline dates (up to 4 weeks)
    /// </summary>
    public async Task<List<string>> GetAvailableBaselinesAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            if (!Directory.Exists(_baselineDirectory))
            {
                return new List<string>();
            }
            
            var baselineDirs = Directory.GetDirectories(_baselineDirectory)
                .Select(Path.GetFileName)
                .Where(d => !string.IsNullOrEmpty(d))
                .Cast<string>()
                .OrderByDescending(d => d)
                .Take(MaxBaselineWeeks)
                .ToList();
            
            _logger.LogInformation("[BASELINE-MGR] Found {Count} available baselines", baselineDirs.Count);
            
            await Task.CompletedTask.ConfigureAwait(false);
            return baselineDirs;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[BASELINE-MGR] Failed to get available baselines");
            return new List<string>();
        }
    }
    
    /// <summary>
    /// Get metadata for specific baseline
    /// </summary>
    public async Task<BaselineMetadata?> GetBaselineMetadataAsync(
        string date,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var baselinePath = Path.Combine(_baselineDirectory, date);
            var metadataPath = Path.Combine(baselinePath, "metadata.json");
            
            if (!File.Exists(metadataPath))
            {
                _logger.LogWarning("[BASELINE-MGR] Metadata file not found for baseline {Date}", date);
                return null;
            }
            
            var json = await File.ReadAllTextAsync(metadataPath, cancellationToken).ConfigureAwait(false);
            var metadata = JsonSerializer.Deserialize<BaselineMetadata>(json);
            
            return metadata;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[BASELINE-MGR] Failed to load metadata for baseline {Date}", date);
            return null;
        }
    }
    
    /// <summary>
    /// Cleanup baselines older than 4 weeks
    /// Maintains rolling window of recent baselines
    /// </summary>
    public async Task CleanupOldBaselinesAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            if (!Directory.Exists(_baselineDirectory))
            {
                return;
            }
            
            var baselineDirs = Directory.GetDirectories(_baselineDirectory)
                .Select(d => new { Path = d, Name = Path.GetFileName(d) })
                .Where(d => !string.IsNullOrEmpty(d.Name))
                .OrderByDescending(d => d.Name)
                .ToList();
            
            // Keep only the most recent MaxBaselineWeeks baselines
            var toDelete = baselineDirs.Skip(MaxBaselineWeeks).ToList();
            
            foreach (var dir in toDelete)
            {
                try
                {
                    Directory.Delete(dir.Path, recursive: true);
                    _logger.LogInformation("[BASELINE-MGR] Deleted old baseline: {Name}", dir.Name);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[BASELINE-MGR] Failed to delete baseline: {Name}", dir.Name);
                }
            }
            
            if (toDelete.Any())
            {
                _logger.LogInformation("[BASELINE-MGR] Cleaned up {Count} old baselines", toDelete.Count);
            }
            
            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[BASELINE-MGR] Failed to cleanup old baselines");
        }
    }
    
    /// <summary>
    /// Check if baseline exists for specific date
    /// </summary>
    public bool BaselineExists(string date)
    {
        var baselinePath = Path.Combine(_baselineDirectory, date);
        return Directory.Exists(baselinePath) && 
               Directory.GetFiles(baselinePath, "*.onnx").Any();
    }
}

/// <summary>
/// Metadata stored with each baseline snapshot
/// </summary>
public sealed class BaselineMetadata
{
    [JsonPropertyName("captureDate")]
    public DateTime CaptureDate { get; set; }
    
    [JsonPropertyName("modelCount")]
    public int ModelCount { get; set; }
    
    [JsonPropertyName("performanceMetrics")]
    public Dictionary<string, decimal> PerformanceMetrics { get; set; } = new();
    
    [JsonPropertyName("promotionTimestamp")]
    public DateTime PromotionTimestamp { get; set; }
    
    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;
    
    [JsonPropertyName("notes")]
    public string? Notes { get; set; }
}
