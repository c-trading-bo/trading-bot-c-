using Microsoft.Extensions.Logging;
using System;
using System.IO;
<<<<<<< HEAD
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.Security.Cryptography;
using System.Text;
=======
using System.Threading;
using System.Threading.Tasks;
>>>>>>> origin/main

namespace BotCore;

/// <summary>
<<<<<<< HEAD
/// Simple cloud RL trainer that checks for model updates periodically
=======
/// Cloud-enabled RL trainer: Local execution + Cloud learning
/// Uploads training data to cloud, downloads improved models automatically
>>>>>>> origin/main
/// </summary>
public sealed class CloudRlTrainer : IDisposable
{
    private readonly ILogger _log;
<<<<<<< HEAD
    private readonly HttpClient _http;
    private readonly string _dataDir;
    private readonly string _modelDir;
    private readonly Timer _timer;
    private bool _disposed;

    public CloudRlTrainer(ILogger logger, HttpClient? httpClient = null)
    {
        _log = logger;
        _http = httpClient ?? new HttpClient();
        _dataDir = Path.Combine(AppContext.BaseDirectory, "data", "rl_training");
        _modelDir = Path.Combine(AppContext.BaseDirectory, "models", "rl");

        Directory.CreateDirectory(_dataDir);
        Directory.CreateDirectory(_modelDir);

        // Check for updates every 30 minutes
        var pollInterval = TimeSpan.FromMinutes(30);
        _timer = new Timer(CheckForUpdatesCallback, null, TimeSpan.Zero, pollInterval);
        _log.LogInformation("[CloudRlTrainer] Started - checking for updates every {Interval}", pollInterval);
    }

    private async void CheckForUpdatesCallback(object? state)
    {
        try
        {
            _log.LogInformation("[CloudRlTrainer] Checking GitHub Releases for new models...");
            await CheckGitHubReleasesAsync();
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[CloudRlTrainer] Error checking for updates");
        }
    }

    private async Task CheckGitHubReleasesAsync()
    {
        try
        {
            // Check GitHub API for latest release
            var apiUrl = "https://api.github.com/repos/kevinsuero072897-collab/trading-bot-c-/releases/latest";
            _http.DefaultRequestHeaders.UserAgent.ParseAdd("TradingBot/1.0");
            
            var response = await _http.GetStringAsync(apiUrl);
            var release = JsonSerializer.Deserialize<JsonElement>(response);
            
            if (release.TryGetProperty("tag_name", out var tagElement))
            {
                var latestTag = tagElement.GetString();
                var lastCheckFile = Path.Combine(_modelDir, "last_github_check.txt");
                
                string? lastCheckedTag = null;
                if (File.Exists(lastCheckFile))
                {
                    lastCheckedTag = await File.ReadAllTextAsync(lastCheckFile);
                }
                
                if (latestTag != lastCheckedTag)
                {
                    _log.LogInformation("[CloudRlTrainer] New models available: {Tag}", latestTag);
                    await DownloadLatestModelsAsync(release);
                    await File.WriteAllTextAsync(lastCheckFile, latestTag);
                }
                else
                {
                    _log.LogDebug("[CloudRlTrainer] Models up to date: {Tag}", latestTag);
=======
    private readonly Timer _timer;
    private readonly string _dataDir;
    private readonly string _modelDir;
    private readonly string _cloudBucket;
    private bool _disposed;

    public CloudRlTrainer(ILogger logger, string cloudBucket = "")
    {
        _log = logger;
        _dataDir = Path.Combine(AppContext.BaseDirectory, "data", "rl_training");
        _modelDir = Path.Combine(AppContext.BaseDirectory, "models", "rl");
        _cloudBucket = cloudBucket;
        
        Directory.CreateDirectory(_dataDir);
        Directory.CreateDirectory(_modelDir);
        
        // Check every 2 hours for cloud sync
        _timer = new Timer(SyncWithCloud, null, TimeSpan.Zero, TimeSpan.FromHours(2));
        _log.LogInformation("[CloudRlTrainer] Started - syncing with cloud every 2 hours");
    }

    private async void SyncWithCloud(object? state)
    {
        if (_disposed) return;

        try
        {
            await SyncWithCloudAsync();
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[CloudRlTrainer] Cloud sync failed");
        }
    }

    private async Task SyncWithCloudAsync()
    {
        // 1. Upload training data to cloud
        await UploadTrainingDataAsync();
        
        // 2. Check for improved models from cloud
        await DownloadImprovedModelsAsync();
    }

    private async Task UploadTrainingDataAsync()
    {
        try
        {
            if (string.IsNullOrEmpty(_cloudBucket))
            {
                _log.LogDebug("[CloudRlTrainer] No cloud bucket configured - skipping upload");
                return;
            }

            // Check if we have new training data to upload
            var dataFiles = Directory.GetFiles(_dataDir, "*.jsonl");
            if (dataFiles.Length == 0)
            {
                _log.LogDebug("[CloudRlTrainer] No training data files found");
                return;
            }

            // Upload using your preferred cloud provider
            await UploadToCloudProvider();
            
            _log.LogInformation("[CloudRlTrainer] Uploaded {FileCount} training files to cloud", dataFiles.Length);
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[CloudRlTrainer] Failed to upload training data");
        }
    }

    private async Task UploadToCloudProvider()
    {
        // AWS S3 Example
        var awsCommand = $"aws s3 sync \"{_dataDir}\" s3://{_cloudBucket}/training-data/ --exclude \"*.tmp\"";
        await RunCloudCommand(awsCommand, "AWS S3 upload");

        // Alternative: Azure Blob
        // var azCommand = $"az storage blob upload-batch --destination {_cloudBucket} --source \"{_dataDir}\"";
        
        // Alternative: Google Cloud Storage  
        // var gcpCommand = $"gsutil -m rsync -r \"{_dataDir}\" gs://{_cloudBucket}/training-data/";
    }

    private async Task DownloadImprovedModelsAsync()
    {
        try
        {
            if (string.IsNullOrEmpty(_cloudBucket))
            {
                _log.LogDebug("[CloudRlTrainer] No cloud bucket configured - skipping download");
                return;
            }

            // Download latest models from cloud
            await DownloadFromCloudProvider();
            
            // Check if we got a new model
            var latestModel = Path.Combine(_modelDir, "latest_rl_sizer.onnx");
            if (File.Exists(latestModel))
            {
                var lastWrite = File.GetLastWriteTimeUtc(latestModel);
                if (DateTime.UtcNow - lastWrite < TimeSpan.FromHours(3))
                {
                    _log.LogInformation("[CloudRlTrainer] 🚀 Downloaded improved model from cloud! Age: {Age:F1} hours", 
                        (DateTime.UtcNow - lastWrite).TotalHours);
>>>>>>> origin/main
                }
            }
        }
        catch (Exception ex)
        {
<<<<<<< HEAD
            _log.LogWarning(ex, "[CloudRlTrainer] Failed to check GitHub releases");
        }
    }

    private async Task DownloadLatestModelsAsync(JsonElement release)
    {
        try
        {
            if (!release.TryGetProperty("assets", out var assetsElement)) return;
            
            foreach (var asset in assetsElement.EnumerateArray())
            {
                if (!asset.TryGetProperty("name", out var nameElement)) continue;
                var assetName = nameElement.GetString();
                
                if (assetName?.EndsWith(".tar.gz") == true)
                {
                    if (!asset.TryGetProperty("browser_download_url", out var urlElement)) continue;
                    var downloadUrl = urlElement.GetString();
                    
                    _log.LogInformation("[CloudRlTrainer] Downloading models: {AssetName}", assetName);
                    
                    var modelBytes = await _http.GetByteArrayAsync(downloadUrl);
                    var tempFile = Path.Combine(_modelDir, assetName);
                    await File.WriteAllBytesAsync(tempFile, modelBytes);
                    
                    // Extract the tar.gz file (simplified - in production use a proper library)
                    _log.LogInformation("[CloudRlTrainer] Downloaded {Bytes} bytes to {File}", modelBytes.Length, tempFile);
                    break;
                }
=======
            _log.LogError(ex, "[CloudRlTrainer] Failed to download models from cloud");
        }
    }

    private async Task DownloadFromCloudProvider()
    {
        // AWS S3 Example
        var awsCommand = $"aws s3 sync s3://{_cloudBucket}/models/ \"{_modelDir}\" --exclude \"*.tmp\"";
        await RunCloudCommand(awsCommand, "AWS S3 download");

        // Alternative: Azure Blob
        // var azCommand = $"az storage blob download-batch --destination \"{_modelDir}\" --source {_cloudBucket}/models";
        
        // Alternative: Google Cloud Storage
        // var gcpCommand = $"gsutil -m rsync -r gs://{_cloudBucket}/models/ \"{_modelDir}\"";
    }

    private async Task RunCloudCommand(string command, string description)
    {
        try
        {
            var process = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    Arguments = $"/c {command}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                }
            };

            process.Start();
            var output = await process.StandardOutput.ReadToEndAsync();
            var error = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (process.ExitCode == 0)
            {
                _log.LogDebug("[CloudRlTrainer] {Description} successful", description);
            }
            else
            {
                _log.LogWarning("[CloudRlTrainer] {Description} failed: {Error}", description, error);
>>>>>>> origin/main
            }
        }
        catch (Exception ex)
        {
<<<<<<< HEAD
            _log.LogError(ex, "[CloudRlTrainer] Failed to download models");
=======
            _log.LogError(ex, "[CloudRlTrainer] Failed to run cloud command: {Command}", command);
>>>>>>> origin/main
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        
        _timer?.Dispose();
<<<<<<< HEAD
        _http?.Dispose();
        _log.LogInformation("[CloudRlTrainer] Disposed");
    }
}
=======
        _log.LogInformation("[CloudRlTrainer] Stopped");
    }
}
>>>>>>> origin/main
