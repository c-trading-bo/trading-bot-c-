using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Octokit;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// GitHub Backup Service - Optional cloud backup of training artifacts
/// Phase 11: GitHub Cloud Backup System
/// 
/// Provides backup of manifests and summaries to GitHub repository for:
/// - Auditability and compliance tracking
/// - Disaster recovery of training metadata
/// - Historical training session records
/// 
/// Note: Full model files (4-10GB) are NOT backed up to GitHub.
/// Models are archived locally for rollback and disaster recovery.
/// </summary>
internal sealed class GitHubBackupService
{
    private readonly ILogger<GitHubBackupService> _logger;
    private readonly IConfiguration _configuration;
    private readonly string _backupHistoryPath;
    private GitHubClient? _githubClient;
    private string? _owner;
    private string? _repo;
    private string _branch = "main";
    private bool _enabled = false;
    private const int MaxRetries = 3;

    public GitHubBackupService(
        ILogger<GitHubBackupService> logger,
        IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
        _backupHistoryPath = Path.Combine(
            Directory.GetCurrentDirectory(), 
            "artifacts", 
            "backups", 
            "backup-history.json");

        InitializeGitHubClient();
    }

    /// <summary>
    /// Initialize GitHub API client with configuration
    /// </summary>
    private void InitializeGitHubClient()
    {
        try
        {
            // Read configuration
            var token = _configuration["GitHub:BackupToken"] 
                ?? Environment.GetEnvironmentVariable("GITHUB_BACKUP_TOKEN");
            _owner = _configuration["GitHub:BackupOwner"] 
                ?? _configuration["GitHub:Owner"];
            _repo = _configuration["GitHub:BackupRepository"] 
                ?? _configuration["GitHub:Repository"];
            _branch = _configuration["GitHub:BackupBranch"] ?? "main";

            if (string.IsNullOrEmpty(token))
            {
                _logger.LogWarning("[GITHUB BACKUP] No GitHub token configured - backups disabled");
                _enabled = false;
                return;
            }

            if (string.IsNullOrEmpty(_owner) || string.IsNullOrEmpty(_repo))
            {
                _logger.LogWarning("[GITHUB BACKUP] GitHub repository not configured - backups disabled");
                _enabled = false;
                return;
            }

            // Initialize Octokit client
            _githubClient = new GitHubClient(new ProductHeaderValue("QBot-Lab-Backup"))
            {
                Credentials = new Credentials(token)
            };

            _enabled = true;
            _logger.LogInformation("[GITHUB BACKUP] Initialized - Owner: {Owner}, Repo: {Repo}, Branch: {Branch}",
                _owner, _repo, _branch);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[GITHUB BACKUP] Failed to initialize: {Error}", ex.Message);
            _enabled = false;
        }
    }

    /// <summary>
    /// Upload training manifest to GitHub repository
    /// Phase 11: Manifest Upload
    /// </summary>
    public async Task<bool> UploadManifestAsync(
        string manifestPath, 
        string sessionId, 
        CancellationToken cancellationToken = default)
    {
        if (!_enabled || _githubClient == null)
        {
            _logger.LogDebug("[GITHUB BACKUP] Skipping manifest upload (disabled)");
            return false;
        }

        try
        {
            _logger.LogInformation("[GITHUB BACKUP] Uploading manifest for session {SessionId}...", sessionId);

            // Read manifest file
            if (!File.Exists(manifestPath))
            {
                _logger.LogError("[GITHUB BACKUP] Manifest file not found: {Path}", manifestPath);
                return false;
            }

            var manifestContent = await File.ReadAllTextAsync(manifestPath, cancellationToken)
                .ConfigureAwait(false);

            // Compress if over 1MB
            byte[] contentBytes;
            var fileName = $"manifest-{sessionId}.json";
            if (manifestContent.Length > 1024 * 1024)
            {
                _logger.LogDebug("[GITHUB BACKUP] Compressing large manifest (>{Size}MB)", 
                    manifestContent.Length / (1024 * 1024));
                contentBytes = CompressString(manifestContent);
                fileName = $"manifest-{sessionId}.json.gz";
            }
            else
            {
                contentBytes = Encoding.UTF8.GetBytes(manifestContent);
            }

            // Upload to GitHub with retry
            var filePath = $"lab-backups/manifests/{fileName}";
            var success = await UploadFileWithRetryAsync(
                filePath,
                contentBytes,
                $"Lab training manifest - {sessionId}",
                cancellationToken).ConfigureAwait(false);

            if (success)
            {
                _logger.LogInformation("[GITHUB BACKUP] ✓ Manifest uploaded: {Path}", filePath);
                await RecordBackupAsync(sessionId, "manifest", filePath, true, cancellationToken)
                    .ConfigureAwait(false);
            }

            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[GITHUB BACKUP] Failed to upload manifest: {Error}", ex.Message);
            await RecordBackupAsync(sessionId, "manifest", "", false, cancellationToken)
                .ConfigureAwait(false);
            return false;
        }
    }

    /// <summary>
    /// Upload training summary to GitHub repository
    /// Phase 11: Summary Upload
    /// </summary>
    public async Task<bool> UploadTrainingSummaryAsync(
        string summaryPath, 
        string sessionId, 
        CancellationToken cancellationToken = default)
    {
        if (!_enabled || _githubClient == null)
        {
            _logger.LogDebug("[GITHUB BACKUP] Skipping summary upload (disabled)");
            return false;
        }

        try
        {
            _logger.LogInformation("[GITHUB BACKUP] Uploading training summary for session {SessionId}...", 
                sessionId);

            // Read or generate summary
            string summaryContent;
            if (File.Exists(summaryPath))
            {
                summaryContent = await File.ReadAllTextAsync(summaryPath, cancellationToken)
                    .ConfigureAwait(false);
            }
            else
            {
                // Generate basic summary if file doesn't exist
                summaryContent = JsonSerializer.Serialize(new
                {
                    SessionId = sessionId,
                    Timestamp = DateTime.UtcNow,
                    Status = "Summary file not found"
                }, new JsonSerializerOptions { WriteIndented = true });
            }

            var contentBytes = Encoding.UTF8.GetBytes(summaryContent);
            var filePath = $"lab-backups/summaries/summary-{sessionId}.json";

            // Upload with retry
            var success = await UploadFileWithRetryAsync(
                filePath,
                contentBytes,
                $"Lab training summary - {sessionId}",
                cancellationToken).ConfigureAwait(false);

            if (success)
            {
                _logger.LogInformation("[GITHUB BACKUP] ✓ Summary uploaded: {Path}", filePath);
                await RecordBackupAsync(sessionId, "summary", filePath, true, cancellationToken)
                    .ConfigureAwait(false);
            }

            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[GITHUB BACKUP] Failed to upload summary: {Error}", ex.Message);
            await RecordBackupAsync(sessionId, "summary", "", false, cancellationToken)
                .ConfigureAwait(false);
            return false;
        }
    }

    /// <summary>
    /// Archive trained models locally (NOT uploaded to GitHub - too large)
    /// Phase 11: Local Model Archiving
    /// </summary>
    public async Task<bool> ArchiveModelsLocallyAsync(
        string modelsPath, 
        string sessionId, 
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("[GITHUB BACKUP] Archiving models locally for session {SessionId}...", 
                sessionId);

            var backupsDir = Path.Combine(Directory.GetCurrentDirectory(), "artifacts", "backups");
            Directory.CreateDirectory(backupsDir);

            var archivePath = Path.Combine(backupsDir, $"models-{sessionId}.zip");

            // Check if models directory exists
            if (!Directory.Exists(modelsPath))
            {
                _logger.LogWarning("[GITHUB BACKUP] Models directory not found: {Path}", modelsPath);
                return false;
            }

            // Create ZIP archive
            if (File.Exists(archivePath))
            {
                File.Delete(archivePath);
            }

            await Task.Run(() => 
                ZipFile.CreateFromDirectory(modelsPath, archivePath, CompressionLevel.Optimal, false),
                cancellationToken).ConfigureAwait(false);

            var archiveInfo = new FileInfo(archivePath);
            var sizeMB = archiveInfo.Length / (1024.0 * 1024.0);

            _logger.LogInformation("[GITHUB BACKUP] ✓ Models archived locally: {Path} ({Size:F1} MB)",
                archivePath, sizeMB);

            // Clean up old archives (keep last 3)
            await CleanupOldArchivesAsync(backupsDir, cancellationToken).ConfigureAwait(false);

            await RecordBackupAsync(sessionId, "models_local", archivePath, true, cancellationToken)
                .ConfigureAwait(false);

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[GITHUB BACKUP] Failed to archive models: {Error}", ex.Message);
            await RecordBackupAsync(sessionId, "models_local", "", false, cancellationToken)
                .ConfigureAwait(false);
            return false;
        }
    }

    /// <summary>
    /// Restore training manifest from GitHub (disaster recovery)
    /// Phase 11: Restore Capability
    /// </summary>
    public async Task<bool> RestoreManifestFromGitHubAsync(
        string? sessionId = null, 
        CancellationToken cancellationToken = default)
    {
        if (!_enabled || _githubClient == null)
        {
            _logger.LogWarning("[GITHUB BACKUP] Cannot restore - GitHub backup not configured");
            return false;
        }

        try
        {
            _logger.LogInformation("[GITHUB BACKUP] Restoring manifest from GitHub...");

            // List all manifests in backup directory
            var contents = await _githubClient.Repository.Content
                .GetAllContents(_owner!, _repo!, "lab-backups/manifests")
                .ConfigureAwait(false);

            if (contents.Count == 0)
            {
                _logger.LogWarning("[GITHUB BACKUP] No manifests found in GitHub backup");
                return false;
            }

            // Find target manifest (specific session or latest)
            var targetFile = sessionId != null
                ? contents.FirstOrDefault(c => c.Name.Contains(sessionId))
                : contents.OrderByDescending(c => c.Name).FirstOrDefault();

            if (targetFile == null)
            {
                _logger.LogWarning("[GITHUB BACKUP] Manifest not found for session: {SessionId}", 
                    sessionId ?? "latest");
                return false;
            }

            // Download manifest
            var fileContent = await _githubClient.Repository.Content
                .GetRawContent(_owner!, _repo!, targetFile.Path)
                .ConfigureAwait(false);

            // Decompress if needed
            string manifestContent;
            if (targetFile.Name.EndsWith(".gz"))
            {
                manifestContent = DecompressString(fileContent);
            }
            else
            {
                manifestContent = Encoding.UTF8.GetString(fileContent);
            }

            // Save to local disk
            var manifestsDir = Path.Combine(Directory.GetCurrentDirectory(), "manifests");
            Directory.CreateDirectory(manifestsDir);
            var localPath = Path.Combine(manifestsDir, targetFile.Name.Replace(".gz", ""));
            await File.WriteAllTextAsync(localPath, manifestContent, cancellationToken)
                .ConfigureAwait(false);

            _logger.LogInformation("[GITHUB BACKUP] ✓ Manifest restored: {Path}", localPath);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[GITHUB BACKUP] Failed to restore manifest: {Error}", ex.Message);
            return false;
        }
    }

    #region Private Helper Methods

    /// <summary>
    /// Upload file to GitHub with exponential backoff retry
    /// </summary>
    private async Task<bool> UploadFileWithRetryAsync(
        string filePath,
        byte[] content,
        string commitMessage,
        CancellationToken cancellationToken)
    {
        for (int attempt = 1; attempt <= MaxRetries; attempt++)
        {
            try
            {
                // Check if file already exists
                RepositoryContent? existingFile = null;
                try
                {
                    var contents = await _githubClient!.Repository.Content
                        .GetAllContents(_owner!, _repo!, filePath)
                        .ConfigureAwait(false);
                    existingFile = contents.FirstOrDefault();
                }
                catch (NotFoundException)
                {
                    // File doesn't exist - this is expected for new files
                }

                // Create or update file
                if (existingFile != null)
                {
                    // Update existing file
                    var updateRequest = new UpdateFileRequest(commitMessage, Convert.ToBase64String(content), existingFile.Sha, _branch);
                    await _githubClient!.Repository.Content
                        .UpdateFile(_owner!, _repo!, filePath, updateRequest)
                        .ConfigureAwait(false);
                }
                else
                {
                    // Create new file
                    var createRequest = new CreateFileRequest(commitMessage, Convert.ToBase64String(content), _branch);
                    await _githubClient!.Repository.Content
                        .CreateFile(_owner!, _repo!, filePath, createRequest)
                        .ConfigureAwait(false);
                }

                return true;
            }
            catch (ApiException ex) when (ex.StatusCode == System.Net.HttpStatusCode.Forbidden)
            {
                // Rate limit hit - wait and retry
                _logger.LogWarning("[GITHUB BACKUP] Rate limit hit, waiting before retry {Attempt}/{Max}...",
                    attempt, MaxRetries);
                
                var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt));
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[GITHUB BACKUP] Upload attempt {Attempt}/{Max} failed: {Error}",
                    attempt, MaxRetries, ex.Message);

                if (attempt == MaxRetries)
                {
                    return false;
                }

                var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt));
                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
        }

        return false;
    }

    /// <summary>
    /// Clean up old model archives (keep last 3)
    /// </summary>
    private async Task CleanupOldArchivesAsync(string backupsDir, CancellationToken cancellationToken)
    {
        try
        {
            var archives = Directory.GetFiles(backupsDir, "models-*.zip")
                .Select(f => new FileInfo(f))
                .OrderByDescending(f => f.CreationTimeUtc)
                .ToList();

            if (archives.Count > 3)
            {
                var toDelete = archives.Skip(3).ToList();
                _logger.LogInformation("[GITHUB BACKUP] Cleaning up {Count} old archives (keeping last 3)",
                    toDelete.Count);

                foreach (var archive in toDelete)
                {
                    try
                    {
                        archive.Delete();
                        _logger.LogDebug("[GITHUB BACKUP] Deleted old archive: {Name}", archive.Name);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "[GITHUB BACKUP] Failed to delete archive: {Name}", 
                            archive.Name);
                    }
                }
            }

            await Task.CompletedTask.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[GITHUB BACKUP] Failed to cleanup old archives: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Record backup operation in history file
    /// </summary>
    private async Task RecordBackupAsync(
        string sessionId,
        string backupType,
        string path,
        bool success,
        CancellationToken cancellationToken)
    {
        try
        {
            // Ensure directory exists
            Directory.CreateDirectory(Path.GetDirectoryName(_backupHistoryPath)!);

            // Load existing history
            var history = new List<BackupRecord>();
            if (File.Exists(_backupHistoryPath))
            {
                var json = await File.ReadAllTextAsync(_backupHistoryPath, cancellationToken)
                    .ConfigureAwait(false);
                var existing = JsonSerializer.Deserialize<List<BackupRecord>>(json);
                if (existing != null)
                {
                    history = existing;
                }
            }

            // Add new record
            history.Add(new BackupRecord
            {
                SessionId = sessionId,
                Timestamp = DateTime.UtcNow,
                BackupType = backupType,
                Path = path,
                Success = success
            });

            // Keep only last 100 records
            if (history.Count > 100)
            {
                history = history.OrderByDescending(h => h.Timestamp).Take(100).ToList();
            }

            // Save updated history
            var updatedJson = JsonSerializer.Serialize(history, new JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            await File.WriteAllTextAsync(_backupHistoryPath, updatedJson, cancellationToken)
                .ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[GITHUB BACKUP] Failed to record backup history: {Error}", ex.Message);
        }
    }

    /// <summary>
    /// Compress string content to byte array
    /// </summary>
    private static byte[] CompressString(string text)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        using var memoryStream = new MemoryStream();
        using (var gzipStream = new GZipStream(memoryStream, CompressionLevel.Optimal))
        {
            gzipStream.Write(bytes, 0, bytes.Length);
        }
        return memoryStream.ToArray();
    }

    /// <summary>
    /// Decompress byte array to string
    /// </summary>
    private static string DecompressString(byte[] compressedBytes)
    {
        using var memoryStream = new MemoryStream(compressedBytes);
        using var gzipStream = new GZipStream(memoryStream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzipStream, Encoding.UTF8);
        return reader.ReadToEnd();
    }

    #endregion
}

#region Supporting Types

/// <summary>
/// Backup record for history tracking
/// </summary>
internal class BackupRecord
{
    public string SessionId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string BackupType { get; set; } = string.Empty;
    public string Path { get; set; } = string.Empty;
    public bool Success { get; set; }
    public string? GitCommitSha { get; set; }
    public long? BackupSizeBytes { get; set; }
}

#endregion
