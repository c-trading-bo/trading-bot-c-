using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System.Net.Http;
using System.Text.Json;
using System.IO.Compression;
using BotCore.ML;

namespace BotCore.Services;

/// <summary>
/// Production-grade service that automatically synchronizes trained models from GitHub Actions workflows
/// Enhances existing UnifiedTradingBrain by providing fresh cloud-trained models with comprehensive error handling
/// </summary>
public class CloudModelSynchronizationService : BackgroundService
{
    private readonly ILogger<CloudModelSynchronizationService> _logger;
    private readonly HttpClient _httpClient;
    private readonly IMLMemoryManager _memoryManager;
    private readonly ProductionResilienceService? _resilienceService;
    private readonly ProductionMonitoringService? _monitoringService;
    private readonly string _githubToken;
    private readonly string _repositoryOwner;
    private readonly string _repositoryName;
    private readonly string _modelsDirectory;
    private readonly TimeSpan _syncInterval;
    
    // Track model versions and performance
    private readonly Dictionary<string, ModelInfo> _currentModels = new();
    private readonly object _syncLock = new();
    private DateTime _lastSyncTime = DateTime.MinValue;

    public CloudModelSynchronizationService(
        ILogger<CloudModelSynchronizationService> logger,
        HttpClient httpClient,
        IMLMemoryManager memoryManager,
        IConfiguration configuration,
        ProductionResilienceService? resilienceService = null,
        ProductionMonitoringService? monitoringService = null)
    {
        _logger = logger;
        _httpClient = httpClient;
        _memoryManager = memoryManager;
        if (configuration is null) throw new ArgumentNullException(nameof(configuration));
        _resilienceService = resilienceService;
        _monitoringService = monitoringService;
        
        // Configure GitHub API access with validation
        _githubToken = Environment.GetEnvironmentVariable("GITHUB_TOKEN") ?? 
                      configuration["GitHub:Token"] ?? "";
        
        if (string.IsNullOrWhiteSpace(_githubToken))
        {
            _logger.LogWarning("⚠️ [CLOUD-SYNC] GitHub token not configured - cloud model sync will be disabled");
        }
        
        _repositoryOwner = configuration["GitHub:Owner"] ?? "c-trading-bo";
        _repositoryName = configuration["GitHub:Repository"] ?? "trading-bot-c-";
        _modelsDirectory = Path.Combine(Directory.GetCurrentDirectory(), "models");
        _syncInterval = TimeSpan.FromMinutes(int.Parse(configuration["CloudSync:IntervalMinutes"] ?? "15"));
        
        // Configure HTTP client for GitHub API with proper headers
        _httpClient.DefaultRequestHeaders.Clear();
        _httpClient.DefaultRequestHeaders.Add("User-Agent", "TradingBot-CloudSync/1.0");
        _httpClient.Timeout = TimeSpan.FromMinutes(5); // Allow time for large model downloads
        if (!string.IsNullOrEmpty(_githubToken))
        {
            _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {_githubToken}");
        }
        
        // Ensure models directory exists
        Directory.CreateDirectory(_modelsDirectory);
        Directory.CreateDirectory(Path.Combine(_modelsDirectory, "rl"));
        Directory.CreateDirectory(Path.Combine(_modelsDirectory, "cloud"));
        Directory.CreateDirectory(Path.Combine(_modelsDirectory, "ensemble"));
        
        _logger.LogInformation("🌐 [CLOUD-SYNC] Service initialized - Repository: {Owner}/{Repo}, Sync interval: {Interval}", 
            _repositoryOwner, _repositoryName, _syncInterval);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("🌐 [CLOUD-SYNC] Starting automated model synchronization...");
        
        // Initial sync on startup
        await SynchronizeModelsAsync(stoppingToken).ConfigureAwait(false);
        
        // Continue periodic sync
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await Task.Delay(_syncInterval, stoppingToken).ConfigureAwait(false);
                await SynchronizeModelsAsync(stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🌐 [CLOUD-SYNC] Error in sync loop");
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken).ConfigureAwait(false); // Wait before retry
            }
        }
        
        _logger.LogInformation("🌐 [CLOUD-SYNC] Service stopped");
    }

    /// <summary>
    /// Synchronize all models from GitHub Actions artifacts
    /// </summary>
    public async Task SynchronizeModelsAsync(CancellationToken cancellationToken)
    {
        if (string.IsNullOrEmpty(_githubToken))
        {
            _logger.LogWarning("🌐 [CLOUD-SYNC] No GitHub token configured, skipping sync");
            return;
        }

        lock (_syncLock)
        {
            if (DateTime.UtcNow - _lastSyncTime < TimeSpan.FromMinutes(5))
            {
                return; // Rate limiting
            }
            _lastSyncTime = DateTime.UtcNow;
        }

        try
        {
            _logger.LogInformation("🌐 [CLOUD-SYNC] Starting model synchronization...");
            
            // Get completed workflow runs
            var workflowRuns = await GetCompletedWorkflowRunsAsync(cancellationToken).ConfigureAwait(false);
            
            var syncedCount = 0;
            var newModelCount = 0;
            
            foreach (var run in workflowRuns)
            {
                try
                {
                    var artifacts = await GetWorkflowArtifactsAsync(run.Id, cancellationToken).ConfigureAwait(false);
                    
                    foreach (var artifact in artifacts.Where(a => a.Name.Contains("model") || a.Name.Contains("onnx")))
                    {
                        var wasNew = await DownloadAndUpdateModelAsync(artifact, run, cancellationToken).ConfigureAwait(false);
                        syncedCount++;
                        if (wasNew) newModelCount++;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "🌐 [CLOUD-SYNC] Failed to process workflow run {RunId}", run.Id);
                }
            }
            
            // Update model registry after sync
            await UpdateModelRegistryAsync(cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("🌐 [CLOUD-SYNC] Sync completed - {Total} artifacts processed, {New} new models downloaded", 
                syncedCount, newModelCount);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🌐 [CLOUD-SYNC] Model synchronization failed");
        }
    }

    /// <summary>
    /// Get completed workflow runs from GitHub API
    /// </summary>
    private async Task<List<WorkflowRun>> GetCompletedWorkflowRunsAsync(CancellationToken cancellationToken)
    {
        try
        {
            var url = $"https://api.github.com/repos/{_repositoryOwner}/{_repositoryName}/actions/runs?status=completed&per_page=50";
            var response = await _httpClient.GetAsync(url, cancellationToken).ConfigureAwait(false);
            
            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("🌐 [CLOUD-SYNC] GitHub API request failed: {StatusCode}", response.StatusCode);
                return new List<WorkflowRun>();
            }
            
            var content = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            var result = JsonSerializer.Deserialize<GitHubWorkflowRunsResponse>(content, new JsonSerializerOptions 
            { 
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower 
            });
            
            // Filter for ML training workflows
            var mlWorkflows = result?.WorkflowRuns?.Where(r => 
                r.Name.Contains("train") || r.Name.Contains("ml") || r.Name.Contains("rl") ||
                r.WorkflowId == 0 || // Include all if we can't filter
                r.Conclusion == "success"
            ).ToList() ?? new List<WorkflowRun>();
            
            _logger.LogDebug("🌐 [CLOUD-SYNC] Found {Count} completed ML workflow runs", mlWorkflows.Count);
            return mlWorkflows;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🌐 [CLOUD-SYNC] Failed to get workflow runs");
            return new List<WorkflowRun>();
        }
    }

    /// <summary>
    /// Get artifacts for a specific workflow run
    /// </summary>
    private async Task<List<Artifact>> GetWorkflowArtifactsAsync(long runId, CancellationToken cancellationToken)
    {
        try
        {
            var url = $"https://api.github.com/repos/{_repositoryOwner}/{_repositoryName}/actions/runs/{runId}/artifacts";
            var response = await _httpClient.GetAsync(url, cancellationToken).ConfigureAwait(false);
            
            if (!response.IsSuccessStatusCode)
            {
                return new List<Artifact>();
            }
            
            var content = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            var result = JsonSerializer.Deserialize<GitHubArtifactsResponse>(content, new JsonSerializerOptions 
            { 
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower 
            });
            
            return result?.Artifacts ?? new List<Artifact>();
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "🌐 [CLOUD-SYNC] Failed to get artifacts for run {RunId}", runId);
            return new List<Artifact>();
        }
    }

    /// <summary>
    /// Download and update a model from artifact
    /// </summary>
    private async Task<bool> DownloadAndUpdateModelAsync(Artifact artifact, WorkflowRun run, CancellationToken cancellationToken)
    {
        try
        {
            // Check if we already have this model version
            var modelKey = $"{artifact.Name}_{run.HeadSha[..8]}";
            if (_currentModels.ContainsKey(modelKey))
            {
                return false; // Not new
            }
            
            _logger.LogInformation("🌐 [CLOUD-SYNC] Downloading new model: {Name} from run {RunId}", artifact.Name, run.Id);
            
            // Download artifact
            var downloadUrl = $"https://api.github.com/repos/{_repositoryOwner}/{_repositoryName}/actions/artifacts/{artifact.Id}/zip";
            var downloadResponse = await _httpClient.GetAsync(downloadUrl, cancellationToken).ConfigureAwait(false);
            
            if (!downloadResponse.IsSuccessStatusCode)
            {
                _logger.LogWarning("🌐 [CLOUD-SYNC] Failed to download artifact {ArtifactId}", artifact.Id);
                return false;
            }
            
            // Extract and save model
            using var zipStream = await downloadResponse.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            using var archive = new ZipArchive(zipStream, ZipArchiveMode.Read);
            
            var extracted = false;
            foreach (var entry in archive.Entries)
            {
                if (entry.Name.EndsWith(".onnx") || entry.Name.EndsWith(".pkl") || entry.Name.EndsWith(".json"))
                {
                    var targetPath = DetermineModelPath(artifact.Name, entry.Name);
                    await ExtractAndSaveFileAsync(entry, targetPath, cancellationToken).ConfigureAwait(false);
                    
                    // Update model info
                    _currentModels[modelKey] = new ModelInfo
                    {
                        Name = artifact.Name,
                        Version = run.HeadSha[..8],
                        Path = targetPath,
                        DownloadedAt = DateTime.UtcNow,
                        WorkflowRun = run.Id,
                        Size = entry.Length
                    };
                    
                    extracted = true;
                    _logger.LogInformation("🌐 [CLOUD-SYNC] Model extracted: {Path}", targetPath);
                }
            }
            
            return extracted;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🌐 [CLOUD-SYNC] Failed to download model {ArtifactName}", artifact.Name);
            return false;
        }
    }

    /// <summary>
    /// Determine where to save the model based on artifact name
    /// </summary>
    private string DetermineModelPath(string artifactName, string fileName)
    {
        var lowerName = artifactName.ToLowerInvariant();
        
        if (lowerName.Contains("cvar") || lowerName.Contains("ppo") || lowerName.Contains("rl"))
        {
            return Path.Combine(_modelsDirectory, "rl", fileName);
        }
        else if (lowerName.Contains("ensemble") || lowerName.Contains("blend"))
        {
            return Path.Combine(_modelsDirectory, "ensemble", fileName);
        }
        else if (lowerName.Contains("cloud"))
        {
            return Path.Combine(_modelsDirectory, "cloud", fileName);
        }
        else
        {
            return Path.Combine(_modelsDirectory, fileName);
        }
    }

    /// <summary>
    /// Extract and save file from zip entry
    /// </summary>
    private async Task ExtractAndSaveFileAsync(ZipArchiveEntry entry, string targetPath, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(targetPath)!);
        
        // Create backup if file exists
        if (File.Exists(targetPath))
        {
            var backupPath = targetPath + $".backup_{DateTime.UtcNow:yyyyMMdd_HHmmss}";
            File.Move(targetPath, backupPath);
            _logger.LogDebug("🌐 [CLOUD-SYNC] Created backup: {BackupPath}", backupPath);
        }
        
        using var entryStream = entry.Open();
        using var fileStream = File.Create(targetPath);
        await entryStream.CopyToAsync(fileStream, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Update the model registry with new models
    /// </summary>
    private async Task UpdateModelRegistryAsync(CancellationToken cancellationToken)
    {
        try
        {
            var registryPath = Path.Combine(_modelsDirectory, "model_registry.json");
            var registry = new ModelRegistry
            {
                LastUpdated = DateTime.UtcNow,
                TotalModels = _currentModels.Count
            };
            
            // Add models to the collection property
            foreach (var model in _currentModels.Values)
            {
                registry.Models.Add(model);
            }
            
            var json = JsonSerializer.Serialize(registry, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(registryPath, json, cancellationToken).ConfigureAwait(false);
            
            _logger.LogDebug("🌐 [CLOUD-SYNC] Model registry updated with {Count} models", _currentModels.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🌐 [CLOUD-SYNC] Failed to update model registry");
        }
    }

    /// <summary>
    /// Get current model information for external services
    /// </summary>
    public Dictionary<string, ModelInfo> GetCurrentModels()
    {
        lock (_syncLock)
        {
            return new Dictionary<string, ModelInfo>(_currentModels);
        }
    }

    /// <summary>
    /// Force immediate synchronization
    /// </summary>
    public Task ForceSyncAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🌐 [CLOUD-SYNC] Force sync requested");
        return SynchronizeModelsAsync(cancellationToken);
    }
}

#region Data Models

public class ModelInfo
{
    public string Name { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public string Path { get; set; } = string.Empty;
    public DateTime DownloadedAt { get; set; }
    public long WorkflowRun { get; set; }
    public long Size { get; set; }
}

public class ModelRegistry
{
    public DateTime LastUpdated { get; set; }
    public List<ModelInfo> Models { get; } = new();
    public int TotalModels { get; set; }
}

public class GitHubWorkflowRunsResponse
{
    public List<WorkflowRun> WorkflowRuns { get; } = new();
}

public class WorkflowRun
{
    public long Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string HeadSha { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty;
    public string Conclusion { get; set; } = string.Empty;
    public long WorkflowId { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
}

public class GitHubArtifactsResponse
{
    public List<Artifact> Artifacts { get; } = new();
}

public class Artifact
{
    public long Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public long SizeInBytes { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime ExpiresAt { get; set; }
}

#endregion