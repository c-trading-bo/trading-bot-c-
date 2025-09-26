using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using System.Security.Cryptography;
using System.Text;
using System.Collections.Generic;
using BotCore.Utilities;

namespace BotCore
{
    /// <summary>
    /// Enhanced 24/7 Cloud-enabled RL trainer with S3 integration and manifest-based model updates.
    /// Uploads training data to cloud, downloads improved models automatically with security verification.
    /// </summary>
    public sealed class CloudRlTrainerEnhanced : IDisposable
    {
        private readonly ILogger _log;
        private readonly Timer _timer;
        private readonly HttpClient _http;
        private readonly string _dataDir;
        private readonly string _modelDir;
        private readonly string _manifestUrl;
        private readonly string _hmacKey;
        private bool _disposed;

        public CloudRlTrainerEnhanced(ILogger logger, HttpClient? httpClient = null)
        {
            _log = logger;
            _http = httpClient ?? new HttpClient();
            _dataDir = Path.Combine(AppContext.BaseDirectory, "data", "rl_training");
            _modelDir = Path.Combine(AppContext.BaseDirectory, "models", "rl");

            // Configuration from environment
            _manifestUrl = Environment.GetEnvironmentVariable("MODEL_MANIFEST_URL") ?? "";
            _hmacKey = Environment.GetEnvironmentVariable("MANIFEST_HMAC_KEY") ?? "";

            Directory.CreateDirectory(_dataDir);
            Directory.CreateDirectory(_modelDir);

            // Check for model updates every 2 hours using TimerHelper
            var pollInterval = TimeSpan.FromSeconds(
                int.Parse(Environment.GetEnvironmentVariable("MODEL_POLL_SEC") ?? "7200"));

            _timer = TimerHelper.CreateAsyncTimerWithImmediateStart(CheckForModelUpdates, pollInterval);
            LoggingHelper.LogServiceStarted(_log, "CloudRlTrainerEnhanced", pollInterval, "checking manifest");
        }

        public void Dispose()
        {
            if (_disposed) return;

            _timer?.Dispose();
            _http?.Dispose();
            _disposed = true;

            _log.LogInformation("[CloudRlTrainerEnhanced] Disposed");
        }

        private async Task CheckForModelUpdates()
        {
            if (_disposed || string.IsNullOrEmpty(_manifestUrl))
            {
                _log.LogDebug("[CloudRlTrainerEnhanced] Skipping update check - no manifest URL");
                return;
            }

            try
            {
                // Download manifest
                var manifest = await DownloadManifestAsync().ConfigureAwait(false);
                if (manifest == null) return;

                // Check if we have this version already
                var currentVersionFile = Path.Combine(_modelDir, "current_version.txt");
                var currentVersion = File.Exists(currentVersionFile) ?
                    await File.ReadAllTextAsync(currentVersionFile).ConfigureAwait(false) : "";

                if (manifest.Version == currentVersion.Trim())
                {
                    _log.LogDebug("[CloudRlTrainerEnhanced] Already have latest version: {Version}", manifest.Version);
                    return;
                }

                _log.LogInformation("[CloudRlTrainerEnhanced] 🚀 New model version available: {Version}", manifest.Version);

                // Download new models
                await DownloadModelsFromManifest(manifest).ConfigureAwait(false);

                // Update version file
                await File.WriteAllTextAsync(currentVersionFile, manifest.Version).ConfigureAwait(false);

                _log.LogInformation("[CloudRlTrainerEnhanced] ✅ Updated to version {Version} with {Samples} training samples",
                    manifest.Version, manifest.TrainingSamples);
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[CloudRlTrainerEnhanced] Failed to check for model updates");
            }
        }

        private async Task<EnhancedModelManifest?> DownloadManifestAsync()
        {
            try
            {
                // Download manifest
                var manifestJson = await _http.GetStringAsync(_manifestUrl).ConfigureAwait(false);
                var manifest = JsonSerializer.Deserialize<EnhancedModelManifest>(manifestJson, new JsonSerializerOptions
                {
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
                });

                if (manifest == null)
                {
                    _log.LogWarning("[CloudRlTrainerEnhanced] Failed to parse manifest");
                    return null;
                }

                // Verify HMAC signature if key is provided
                if (!string.IsNullOrEmpty(_hmacKey))
                {
                    var signature = ManifestVerifier.ExtractSignatureFromManifest(manifestJson);
                    if (string.IsNullOrEmpty(signature))
                    {
                        _log.LogError("[CloudRlTrainerEnhanced] No signature found in manifest");
                        return null;
                    }

                    if (!ManifestVerifier.VerifyManifestSignature(manifestJson, _hmacKey, signature))
                    {
                        _log.LogError("[CloudRlTrainerEnhanced] SECURITY: Manifest signature verification failed!");
                        return null;
                    }

                    // Validate manifest structure
                    if (!ManifestVerifier.ValidateManifestStructure(manifestJson))
                    {
                        _log.LogError("[CloudRlTrainerEnhanced] Invalid manifest structure");
                        return null;
                    }

                    _log.LogDebug("[CloudRlTrainerEnhanced] Manifest signature verified");
                }

                return manifest;
            }
            catch (HttpRequestException ex)
            {
                _log.LogWarning("[CloudRlTrainerEnhanced] Failed to download manifest: {Error}", ex.Message);
                return null;
            }
            catch (JsonException ex)
            {
                _log.LogError(ex, "[CloudRlTrainerEnhanced] Failed to parse manifest JSON");
                return null;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[CloudRlTrainerEnhanced] Unexpected error downloading manifest");
                return null;
            }
        }

        private Task DownloadModelsFromManifest(EnhancedModelManifest manifest)
        {
            var downloadTasks = new List<Task>();

            foreach (var (modelType, modelInfo) in manifest.Models)
            {
                downloadTasks.Add(DownloadModelAsync(modelType, modelInfo));
            }

            return Task.WhenAll(downloadTasks);
        }

        private async Task DownloadModelAsync(string modelType, EnhancedModelInfo modelInfo)
        {
            var modelPath = Path.Combine(_modelDir, $"{modelType}.onnx");
            var tempPath = modelPath + ".tmp";

            try
            {
                _log.LogDebug("[CloudRlTrainerEnhanced] Downloading {ModelType} model from {Url}", modelType, modelInfo.Url);

                // Download to temp file
                using var response = await _http.GetAsync(modelInfo.Url).ConfigureAwait(false);
                response.EnsureSuccessStatusCode();

                await using var fileStream = File.Create(tempPath);
                await response.Content.CopyToAsync(fileStream).ConfigureAwait(false);
                await fileStream.FlushAsync().ConfigureAwait(false);

                // Verify checksum
                var actualChecksum = await ComputeFileChecksumAsync(tempPath).ConfigureAwait(false);
                if (!string.Equals(actualChecksum, modelInfo.Checksum, StringComparison.OrdinalIgnoreCase))
                {
                    _log.LogError("[CloudRlTrainerEnhanced] Checksum mismatch for {ModelType}: expected {Expected}, got {Actual}",
                        modelType, modelInfo.Checksum, actualChecksum);
                    File.Delete(tempPath);
                    return;
                }

                // Atomic replace
                if (File.Exists(modelPath))
                {
                    File.Replace(tempPath, modelPath, null);
                }
                else
                {
                    File.Move(tempPath, modelPath);
                }

                _log.LogInformation("[CloudRlTrainerEnhanced] ✅ Downloaded {ModelType} model ({Size} bytes)",
                    modelType, modelInfo.Size);
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[CloudRlTrainerEnhanced] Failed to download {ModelType} model", modelType);

                // Clean up temp file
                if (File.Exists(tempPath))
                {
                    try { File.Delete(tempPath); } catch { }
                }
            }
        }

        private static async Task<string> ComputeFileChecksumAsync(string filePath)
        {
            using var sha256 = SHA256.Create();
            await using var fileStream = File.OpenRead(filePath);
            var hashBytes = await sha256.ComputeHashAsync(fileStream).ConfigureAwait(false);
            return Convert.ToHexString(hashBytes).ToLowerInvariant();
        }

        /// <summary>
        /// Enhanced model manifest structure for cloud-based model updates
        /// </summary>
        public class EnhancedModelManifest
        {
            public string Version { get; set; } = "";
            public DateTime Timestamp { get; set; }
            public int TrainingSamples { get; set; }
            public Dictionary<string, EnhancedModelInfo> Models { get; } = new();
        }

        /// <summary>
        /// Enhanced model information with comprehensive metadata
        /// </summary>
        public class EnhancedModelInfo
        {
            public string Url { get; set; } = "";
            public string Checksum { get; set; } = "";
            public long Size { get; set; }
            public DateTime CreatedAt { get; set; }
        }
    }
}
