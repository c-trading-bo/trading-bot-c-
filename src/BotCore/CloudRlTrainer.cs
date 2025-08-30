using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore;

/// <summary>
/// Cloud-enabled RL trainer: Local execution + Cloud learning
/// Uploads training data to cloud, downloads improved models automatically
/// </summary>
public sealed class CloudRlTrainer : IDisposable
{
    private readonly ILogger _log;
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
                }
            }
        }
        catch (Exception ex)
        {
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
            }
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "[CloudRlTrainer] Failed to run cloud command: {Command}", command);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        
        _timer?.Dispose();
        _log.LogInformation("[CloudRlTrainer] Stopped");
    }
}
