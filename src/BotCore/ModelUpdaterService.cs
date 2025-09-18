using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace BotCore
{
    /// <summary>
    /// Background service that polls for model updates and hot-swaps ONNX models.
    /// Only updates when bot is flat (no active positions) for safety.
    /// </summary>
    public sealed class ModelUpdaterService
    {
        private readonly ILogger<ModelUpdaterService> _log;
        private readonly HttpClient _http;
        private readonly IConfiguration _config;
        private readonly IPositionAgent _positionAgent; // Assume this exists

        private readonly string _manifestUrl;
        private readonly string _hmacKey;
        private readonly int _pollIntervalSeconds;
        private readonly string _modelsPath;

        private string _currentManifestHash = "";
        private DateTime _lastSuccessfulCheck = DateTime.MinValue;
        private int _consecutiveFailures = 0;

        private readonly Timer _updateTimer;
        private readonly object _updateLock = new();
        private bool _isRunning = false;

        public ModelUpdaterService(
            ILogger<ModelUpdaterService> logger,
            HttpClient httpClient,
            IConfiguration configuration,
            IPositionAgent positionAgent)
        {
            _log = logger;
            _http = httpClient;
            _config = configuration;
            _positionAgent = positionAgent;

            // Configuration from appsettings.json or environment
            _manifestUrl = _config["ModelUpdater:ManifestUrl"] ??
                          Environment.GetEnvironmentVariable("MODEL_MANIFEST_URL") ??
                          "https://local.trading-bot.dev/models/current.json";  // Local endpoint only

            _hmacKey = _config["ModelUpdater:HmacKey"] ??
                      Environment.GetEnvironmentVariable("MANIFEST_HMAC_KEY") ??
                      throw new InvalidOperationException("MANIFEST_HMAC_KEY required");

            _pollIntervalSeconds = int.Parse(_config["ModelUpdater:PollIntervalSeconds"] ?? "300"); // 5 minutes default
            _modelsPath = Path.Combine("models", "onnx");

            Directory.CreateDirectory(_modelsPath);

            // Create timer for periodic updates
            _updateTimer = new Timer(OnTimerElapsed, null, TimeSpan.Zero, TimeSpan.FromSeconds(_pollIntervalSeconds));
        }

        public void Start()
        {
            _log.LogInformation("[ModelUpdater] Starting with manifest URL: {Url}, poll interval: {Interval}s",
                _manifestUrl, _pollIntervalSeconds);

            _isRunning = true;
        }

        public void Stop()
        {
            _isRunning = false;
            _updateTimer?.Dispose();
            _log.LogInformation("[ModelUpdater] Service stopped");
        }

        private async void OnTimerElapsed(object? state)
        {
            if (!_isRunning)
                return;

            lock (_updateLock)
            {
                if (!_isRunning)
                    return;
            }

            try
            {
                await CheckForModelUpdates(CancellationToken.None).ConfigureAwait(false);
                _consecutiveFailures = 0;
                _lastSuccessfulCheck = DateTime.UtcNow;
            }
            catch (Exception ex)
            {
                _consecutiveFailures++;
                _log.LogError(ex, "[ModelUpdater] Check failed (attempt {Attempt})", _consecutiveFailures);

                // Exponential backoff on repeated failures
                if (_consecutiveFailures >= 3)
                {
                    var backoffMinutes = Math.Min(_consecutiveFailures * 2, 30);
                    _log.LogWarning("[ModelUpdater] Multiple failures, backing off for {Minutes} minutes", backoffMinutes);

                    // Adjust timer interval for backoff
                    _updateTimer?.Change(TimeSpan.FromMinutes(backoffMinutes), TimeSpan.FromSeconds(_pollIntervalSeconds));
                    return;
                }
            }
        }

        private async Task CheckForModelUpdates(CancellationToken cancellationToken)
        {
            _log.LogDebug("[ModelUpdater] Checking for model updates...");

            // Download and verify manifest
            var manifestJson = await DownloadManifestAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            if (string.IsNullOrEmpty(manifestJson))
            {
                _log.LogWarning("[ModelUpdater] Failed to download manifest");
                return;
            }

            // Check if manifest has changed
            var manifestHash = ComputeHash(manifestJson);
            if (manifestHash == _currentManifestHash)
            {
                _log.LogDebug("[ModelUpdater] No manifest changes detected");
                return;
            }

            // Verify HMAC signature
            if (!VerifyManifestSignature(manifestJson))
            {
                _log.LogError("[ModelUpdater] SECURITY: Manifest signature verification failed!");
                return;
            }

            // Parse manifest
            var manifest = ParseManifest(manifestJson);
            if (manifest == null)
            {
                _log.LogError("[ModelUpdater] Failed to parse manifest");
                return;
            }

            // Safety check: only update when flat
            if (!await IsPositionFlat(cancellationToken))
            {
                _log.LogInformation("[ModelUpdater] Skipping update - active positions detected").ConfigureAwait(false);
                return;
            }

            // Download and install new models
            var updateSuccess = await UpdateModelsAsync(manifest, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
            if (updateSuccess)
            {
                _currentManifestHash = manifestHash;
                _log.LogInformation("[ModelUpdater] Successfully updated to manifest version {Version}",
                    manifest.Version);
            }
        }

        private async Task<string?> DownloadManifestAsync(CancellationToken cancellationToken)
        {
            try
            {
                using var response = await _http.GetAsync(_manifestUrl, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                response.EnsureSuccessStatusCode();

                var content = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                _log.LogDebug("[ModelUpdater] Downloaded manifest ({Bytes} bytes)", content.Length);

                return content;
            }
            catch (HttpRequestException ex)
            {
                _log.LogError(ex, "[ModelUpdater] HTTP error downloading manifest");
                return null;
            }
        }

        private bool VerifyManifestSignature(string manifestJson)
        {
            try
            {
                // Extract signature from manifest
                var signature = ManifestVerifier.ExtractSignatureFromManifest(manifestJson);
                if (string.IsNullOrEmpty(signature))
                {
                    _log.LogError("[ModelUpdater] No signature found in manifest");
                    return false;
                }

                // Verify using ManifestVerifier
                var isValid = ManifestVerifier.VerifyManifestSignature(manifestJson, _hmacKey, signature);

                if (!isValid)
                {
                    _log.LogError("[ModelUpdater] SECURITY: Invalid manifest signature!");
                }
                else
                {
                    _log.LogDebug("[ModelUpdater] Manifest signature verified");
                }

                return isValid;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[ModelUpdater] Signature verification error");
                return false;
            }
        }

        private ModelManifest? ParseManifest(string manifestJson)
        {
            try
            {
                var manifest = JsonSerializer.Deserialize<ModelManifest>(manifestJson, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
                    PropertyNameCaseInsensitive = true
                });

                // Validate required fields
                if (!ManifestVerifier.ValidateManifestStructure(manifestJson))
                {
                    return null;
                }

                return manifest;
            }
            catch (JsonException ex)
            {
                _log.LogError(ex, "[ModelUpdater] Failed to parse manifest JSON");
                return null;
            }
        }

        private async Task<bool> IsPositionFlat(CancellationToken cancellationToken)
        {
            try
            {
                // This would depend on your position agent implementation
                // For now, assume we have a method to check if any positions are open
                var hasPositions = await _positionAgent.HasActivePositionsAsync(cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                return !hasPositions;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[ModelUpdater] Error checking position status");
                return false; // Fail safe - don't update if we can't verify
            }
        }

        private async Task<bool> UpdateModelsAsync(ModelManifest manifest, CancellationToken cancellationToken)
        {
            var allSuccessful = true;

            foreach (var (modelName, modelInfo) in manifest.Models)
            {
                try
                {
                    var success = await UpdateSingleModelAsync(modelName, modelInfo, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                    if (!success)
                    {
                        allSuccessful = false;
                        _log.LogError("[ModelUpdater] Failed to update model: {ModelName}", modelName);
                    }
                    else
                    {
                        _log.LogInformation("[ModelUpdater] Updated model: {ModelName} -> {Checksum}",
                            modelName, modelInfo.Checksum[..8]);
                    }
                }
                catch (Exception ex)
                {
                    _log.LogError(ex, "[ModelUpdater] Error updating model {ModelName}", modelName);
                    allSuccessful = false;
                }
            }

            return allSuccessful;
        }

        private async Task<bool> UpdateSingleModelAsync(string modelName, ModelInfo modelInfo, CancellationToken cancellationToken)
        {
            var tempPath = Path.Combine(_modelsPath, $"{modelName}.tmp");
            var finalPath = Path.Combine(_modelsPath, $"{modelName}.onnx");

            try
            {
                // Download model to temporary file
                using var response = await _http.GetAsync(modelInfo.Url, cancellationToken).ConfigureAwait(false).ConfigureAwait(false);
                response.EnsureSuccessStatusCode();

                await using var fileStream = File.Create(tempPath);
                await response.Content.CopyToAsync(fileStream, cancellationToken).ConfigureAwait(false);
                await fileStream.FlushAsync(cancellationToken).ConfigureAwait(false);
                fileStream.Close();

                // Verify checksum
                var actualChecksum = await ComputeFileChecksumAsync(tempPath).ConfigureAwait(false).ConfigureAwait(false);
                if (!string.Equals(actualChecksum, modelInfo.Checksum, StringComparison.OrdinalIgnoreCase))
                {
                    _log.LogError("[ModelUpdater] Checksum mismatch for {ModelName}: expected {Expected}, got {Actual}",
                        modelName, modelInfo.Checksum, actualChecksum);
                    File.Delete(tempPath);
                    return false;
                }

                // Atomic replace: move temp file to final location
                if (File.Exists(finalPath))
                {
                    var backupPath = finalPath + ".backup";
                    File.Move(finalPath, backupPath);

                    try
                    {
                        File.Move(tempPath, finalPath);
                        File.Delete(backupPath); // Clean up backup
                    }
                    catch
                    {
                        // Restore backup if move failed
                        if (File.Exists(backupPath))
                        {
                            File.Move(backupPath, finalPath);
                        }
                        throw;
                    }
                }
                else
                {
                    File.Move(tempPath, finalPath);
                }

                return true;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[ModelUpdater] Failed to download/install model {ModelName}", modelName);

                // Clean up temp file
                if (File.Exists(tempPath))
                {
                    try { File.Delete(tempPath); } catch { }
                }

                return false;
            }
        }

        private static string ComputeHash(string input)
        {
            using var sha256 = SHA256.Create();
            var hashBytes = sha256.ComputeHash(System.Text.Encoding.UTF8.GetBytes(input));
            return Convert.ToHexString(hashBytes).ToLowerInvariant();
        }

        private static async Task<string> ComputeFileChecksumAsync(string filePath)
        {
            using var sha256 = SHA256.Create();
            await using var fileStream = File.OpenRead(filePath);
            var hashBytes = await sha256.ComputeHashAsync(fileStream).ConfigureAwait(false).ConfigureAwait(false);
            return Convert.ToHexString(hashBytes).ToLowerInvariant();
        }

        public class ModelManifest
        {
            public string Version { get; set; } = "";
            public DateTime Timestamp { get; set; }
            public Dictionary<string, ModelInfo> Models { get; set; } = new();
        }

        public class ModelInfo
        {
            public string Url { get; set; } = "";
            public string Checksum { get; set; } = "";
            public long Size { get; set; }
            public DateTime CreatedAt { get; set; }
        }
    }

    /// <summary>
    /// Interface for position checking (implement based on your position agent)
    /// </summary>
    public interface IPositionAgent
    {
        Task<bool> HasActivePositionsAsync(CancellationToken cancellationToken = default);
    }
}
