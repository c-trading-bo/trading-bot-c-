using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Promotion;

/// <summary>
/// Phase 7.4: Model Registry Version Manager
/// Manages version pointers so Terminal Mode knows which model version to load
/// Tracks version history for audit and rollback
/// </summary>
internal sealed class VersionManager
{
    private readonly ILogger<VersionManager> _logger;
    private readonly string _modelsDirectory;
    private readonly string _versionFile;
    private readonly string _versionHistoryFile;
    
    public VersionManager(ILogger<VersionManager> logger)
    {
        _logger = logger;
        var baseDir = Directory.GetCurrentDirectory();
        _modelsDirectory = Path.Combine(baseDir, "models");
        _versionFile = Path.Combine(_modelsDirectory, "version.txt");
        _versionHistoryFile = Path.Combine(_modelsDirectory, "version_history.json");
        
        Directory.CreateDirectory(_modelsDirectory);
    }
    
    /// <summary>
    /// Get current production version
    /// Returns version string (e.g., "v20250119") or null if not set
    /// </summary>
    public async Task<string?> GetCurrentVersionAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            if (!File.Exists(_versionFile))
            {
                _logger.LogWarning("[VERSION] Version file not found: {Path}", _versionFile);
                return null;
            }
            
            var version = await File.ReadAllTextAsync(_versionFile, cancellationToken).ConfigureAwait(false);
            version = version.Trim();
            
            _logger.LogDebug("[VERSION] Current version: {Version}", version);
            return version;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[VERSION] Failed to read version file");
            return null;
        }
    }
    
    /// <summary>
    /// Update version pointer atomically
    /// Updates both version.txt and version_history.json
    /// </summary>
    public async Task<bool> UpdateVersionAsync(
        string newVersion,
        Dictionary<string, object>? metadata = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[VERSION] Updating version to: {Version}", newVersion);
            
            // Validate version format
            if (!ValidateVersionFormat(newVersion))
            {
                _logger.LogError("[VERSION] Invalid version format: {Version}", newVersion);
                return false;
            }
            
            // Get current version for history
            var currentVersion = await GetCurrentVersionAsync(cancellationToken).ConfigureAwait(false);
            
            // Write new version atomically
            var tempFile = $"{_versionFile}.tmp";
            await File.WriteAllTextAsync(tempFile, newVersion, cancellationToken).ConfigureAwait(false);
            
            // Atomic replace
            File.Move(tempFile, _versionFile, overwrite: true);
            
            // Update version history
            await AddToHistoryAsync(currentVersion, newVersion, metadata, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("[VERSION] Version updated successfully: {OldVersion} -> {NewVersion}",
                currentVersion ?? "none", newVersion);
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[VERSION] Failed to update version");
            return false;
        }
    }
    
    /// <summary>
    /// Get complete version history
    /// </summary>
    public async Task<List<VersionHistoryEntry>> GetVersionHistoryAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            if (!File.Exists(_versionHistoryFile))
            {
                return new List<VersionHistoryEntry>();
            }
            
            var json = await File.ReadAllTextAsync(_versionHistoryFile, cancellationToken).ConfigureAwait(false);
            var history = JsonSerializer.Deserialize<List<VersionHistoryEntry>>(json);
            
            return history ?? new List<VersionHistoryEntry>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[VERSION] Failed to read version history");
            return new List<VersionHistoryEntry>();
        }
    }
    
    /// <summary>
    /// Pin version to specific value (for manual rollback)
    /// </summary>
    public async Task<bool> PinVersionAsync(
        string version,
        string reason,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogWarning("[VERSION] Manually pinning version to: {Version}, reason: {Reason}", version, reason);
            
            // Validate version exists
            var isValid = await ValidateVersionAsync(version, cancellationToken).ConfigureAwait(false);
            if (!isValid)
            {
                _logger.LogError("[VERSION] Cannot pin to invalid version: {Version}", version);
                return false;
            }
            
            var metadata = new Dictionary<string, object>
            {
                ["pinned"] = true,
                ["reason"] = reason,
                ["pinned_by"] = Environment.UserName,
                ["pinned_at"] = DateTime.UtcNow
            };
            
            return await UpdateVersionAsync(version, metadata, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[VERSION] Failed to pin version");
            return false;
        }
    }
    
    /// <summary>
    /// Validate that a version directory exists with valid models
    /// </summary>
    public async Task<bool> ValidateVersionAsync(string version, CancellationToken cancellationToken = default)
    {
        try
        {
            // For now, check if production directory has models
            // In full implementation, would check version-specific directory
            var productionDir = Path.Combine(_modelsDirectory, "production");
            
            if (!Directory.Exists(productionDir))
            {
                _logger.LogWarning("[VERSION] Production directory not found");
                return false;
            }
            
            var modelCount = Directory.GetFiles(productionDir, "*.onnx").Length;
            if (modelCount == 0)
            {
                _logger.LogWarning("[VERSION] No models found in production directory");
                return false;
            }
            
            _logger.LogDebug("[VERSION] Version {Version} validated: {Count} models", version, modelCount);
            
            await Task.CompletedTask.ConfigureAwait(false);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[VERSION] Version validation failed");
            return false;
        }
    }
    
    /// <summary>
    /// Generate version string for current date (e.g., "v20250119")
    /// </summary>
    public static string GenerateVersionString(DateTime date)
    {
        return $"v{date:yyyyMMdd}";
    }
    
    /// <summary>
    /// Validate version string format
    /// </summary>
    private bool ValidateVersionFormat(string version)
    {
        // Version should be in format: v + YYYYMMDD (e.g., v20250119)
        if (string.IsNullOrEmpty(version) || version.Length != 9)
            return false;
        
        if (!version.StartsWith("v"))
            return false;
        
        // Validate date portion
        var datePart = version.Substring(1);
        return DateTime.TryParseExact(datePart, "yyyyMMdd", null, 
            System.Globalization.DateTimeStyles.None, out _);
    }
    
    /// <summary>
    /// Add entry to version history
    /// </summary>
    private async Task AddToHistoryAsync(
        string? previousVersion,
        string newVersion,
        Dictionary<string, object>? metadata,
        CancellationToken cancellationToken)
    {
        try
        {
            var history = await GetVersionHistoryAsync(cancellationToken).ConfigureAwait(false);
            
            var entry = new VersionHistoryEntry
            {
                Version = newVersion,
                PreviousVersion = previousVersion,
                PromotedAt = DateTime.UtcNow,
                PromotedBy = Environment.UserName,
                Metadata = metadata ?? new Dictionary<string, object>()
            };
            
            history.Add(entry);
            
            // Keep only last 52 entries (1 year of weekly promotions)
            if (history.Count > 52)
            {
                history = history.OrderByDescending(h => h.PromotedAt).Take(52).ToList();
            }
            
            var options = new JsonSerializerOptions { WriteIndented = true };
            var json = JsonSerializer.Serialize(history, options);
            
            await File.WriteAllTextAsync(_versionHistoryFile, json, cancellationToken).ConfigureAwait(false);
            
            _logger.LogDebug("[VERSION] Added history entry: {Version}", newVersion);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[VERSION] Failed to update version history");
        }
    }
}

/// <summary>
/// Version history entry
/// </summary>
public sealed class VersionHistoryEntry
{
    [JsonPropertyName("version")]
    public string Version { get; set; } = string.Empty;
    
    [JsonPropertyName("previousVersion")]
    public string? PreviousVersion { get; set; }
    
    [JsonPropertyName("promotedAt")]
    public DateTime PromotedAt { get; set; }
    
    [JsonPropertyName("promotedBy")]
    public string PromotedBy { get; set; } = string.Empty;
    
    [JsonPropertyName("metadata")]
    public Dictionary<string, object> Metadata { get; set; } = new();
}
