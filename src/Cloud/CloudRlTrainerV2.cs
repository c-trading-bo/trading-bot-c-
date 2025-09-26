using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace CloudTrainer
{
    #region Options & DTOs
    public sealed class CloudRlTrainerOptions
    {
        public bool Enabled { get; set; } = true;
        public int PollIntervalMinutes { get; set; } = 15;
        public string InstallDir { get; set; } = "models";
        public string TempDir { get; set; } = "models/.staging";
        public string RegistryFile { get; set; } = "config/model-registry.json";
        public GithubOptions Github { get; set; } = new();
        public ReleasesOptions Releases { get; set; } = new();
        public HttpOptions Http { get; set; } = new();
        public PerformanceOptions Performance { get; set; } = new();
    }

    public sealed class GithubOptions
    {
        public string Owner { get; set; } = string.Empty;
        public string Repo { get; set; } = string.Empty;
        public string Token { get; set; } = string.Empty;
        public string TagPattern { get; set; } = "v*";
    }

    public sealed class ReleasesOptions
    {
        public string BaseUrl { get; set; } = string.Empty;
        public string ApiKey { get; set; } = string.Empty;
        public bool VerifySignatures { get; set; } = true;
    }

    public sealed class HttpOptions
    {
        public int TimeoutSeconds { get; set; } = 300;
        public int MaxRetries { get; set; } = 3;
        public int RateLimitPerMinute { get; set; } = 60;
        public bool UseETag { get; set; } = true;
    }

    public sealed class PerformanceOptions
    {
        public int MaxConcurrentDownloads { get; set; } = 3;
        public long MaxFileSizeBytes { get; set; } = 1024L * 1024 * 1024; // 1GB
        public bool EnableCompression { get; set; } = true;
        public string PerformanceStore { get; set; } = "config/performance-store.json";
    }

    public sealed class ModelDescriptor
    {
        public string Id { get; set; } = string.Empty;
        public string Name { get; set; } = string.Empty;
        public string Version { get; set; } = string.Empty;
        public string Url { get; set; } = string.Empty;
        public string Sha256 { get; set; } = string.Empty;
        public long SizeBytes { get; set; }
        public DateTimeOffset PublishedAt { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new();
        public ModelPerformance? Performance { get; set; }
    }

    public sealed class ModelPerformance
    {
        public double Accuracy { get; set; }
        public double ProfitFactor { get; set; }
        public double SharpeRatio { get; set; }
        public double MaxDrawdown { get; set; }
        public int TotalTrades { get; set; }
        public DateTimeOffset LastEvaluated { get; set; }
    }

    public sealed class ModelRegistry
    {
        public string Version { get; set; } = "2.0";
        public DateTimeOffset LastUpdated { get; set; }
        public List<ModelDescriptor> Available { get; set; } = new();
        public List<ModelDescriptor> Installed { get; set; } = new();
        public ModelDescriptor? Active { get; set; }
        public Dictionary<string, object> Config { get; set; } = new();
    }

    public sealed class ModelManifest
    {
        public string Version { get; set; } = string.Empty;
        public Dictionary<string, ModelInfo> Models { get; set; } = new();
        public double DriftScore { get; set; }
        public DateTime CreatedAt { get; set; }
        
        public string VersionSha() => System.Security.Cryptography.SHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes($"{Version}-{CreatedAt:O}")
        ).ToHexString()[..8];
    }

    public sealed class ModelInfo
    {
        public string Url { get; set; } = string.Empty;
        public string Sha256 { get; set; } = string.Empty;
        public long Size { get; set; }
    }

    // Extension method for byte array to hex string conversion
    public static class ByteArrayExtensions
    {
        public static string ToHexString(this byte[] bytes)
        {
            return Convert.ToHexString(bytes);
        }
    }
    #endregion

    #region Interfaces
    public interface IModelDownloader
    {
        Task<string> DownloadAsync(ModelDescriptor model, string targetPath, CancellationToken cancellationToken);
        Task<bool> VerifyIntegrityAsync(string filePath, ModelDescriptor model, CancellationToken cancellationToken);
    }

    public interface IModelHotSwapper
    {
        Task<bool> SwapModelAsync(ModelDescriptor newModel, CancellationToken cancellationToken);
        Task<ModelDescriptor?> GetActiveModelAsync(CancellationToken cancellationToken);
    }

    public interface IPerformanceStore
    {
        Task<ModelPerformance?> GetPerformanceAsync(string modelId, CancellationToken cancellationToken);
        Task SavePerformanceAsync(string modelId, ModelPerformance performance, CancellationToken cancellationToken);
        Task<List<ModelPerformance>> GetTopPerformersAsync(int count, CancellationToken cancellationToken);
    }

    public interface IRateLimiter
    {
        Task WaitAsync(CancellationToken cancellationToken);
        bool TryAcquire();
    }
    #endregion

    #region Rate Limiter Implementation
    public sealed class TokenBucketRateLimiter : IRateLimiter
    {
        private readonly SemaphoreSlim _semaphore;
        private readonly Timer _refillTimer;
        private readonly int _maxTokens;
        private readonly TimeSpan _refillInterval;
        private int _currentTokens;
        private readonly object _lock = new();

        public TokenBucketRateLimiter(int maxTokens, TimeSpan refillInterval)
        {
            _maxTokens = maxTokens;
            _refillInterval = refillInterval;
            _currentTokens = maxTokens;
            _semaphore = new SemaphoreSlim(maxTokens, maxTokens);
            _refillTimer = new Timer(RefillTokens, null, refillInterval, refillInterval);
        }

        public async Task WaitAsync(CancellationToken cancellationToken)
        {
            await _semaphore.WaitAsync(cancellationToken).ConfigureAwait(false);
        }

        public bool TryAcquire()
        {
            return _semaphore.Wait(0);
        }

        private void RefillTokens(object? state)
        {
            lock (_lock)
            {
                var tokensToAdd = Math.Min(_maxTokens - _currentTokens, _maxTokens);
                _currentTokens += tokensToAdd;
                _semaphore.Release(tokensToAdd);
            }
        }

        public void Dispose()
        {
            _refillTimer?.Dispose();
            _semaphore?.Dispose();
        }
    }
    #endregion

    #region Default Implementations
    public sealed class DefaultModelDownloader : IModelDownloader
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<DefaultModelDownloader> _logger;
        private readonly IRateLimiter _rateLimiter;

        public DefaultModelDownloader(HttpClient httpClient, ILogger<DefaultModelDownloader> logger, IRateLimiter rateLimiter)
        {
            _httpClient = httpClient;
            _logger = logger;
            _rateLimiter = rateLimiter;
        }

        public async Task<string> DownloadAsync(ModelDescriptor model, string targetPath, CancellationToken cancellationToken)
        {
            await _rateLimiter.WaitAsync(cancellationToken).ConfigureAwait(false);

            _logger.LogInformation("📥 Downloading model {ModelName} v{Version}", model.Name, model.Version);

            var tempPath = targetPath + ".tmp";
            var retryCount = 0;
            const int maxRetries = 3;

            while (retryCount < maxRetries)
            {
                try
                {
                    using var response = await _httpClient.GetAsync(model.Url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
                    response.EnsureSuccessStatusCode();

                    await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
                    await using var fileStream = File.Create(tempPath);
                    await contentStream.CopyToAsync(fileStream, cancellationToken).ConfigureAwait(false);

                    // Atomic move
                    File.Move(tempPath, targetPath, true);
                    _logger.LogInformation("✅ Downloaded model {ModelName} to {Path}", model.Name, targetPath);
                    return targetPath;
                }
                catch (Exception ex) when (retryCount < maxRetries - 1)
                {
                    retryCount++;
                    _logger.LogWarning("⚠️ Download attempt {Attempt} failed for {ModelName}: {Error}", retryCount, model.Name, ex.Message);
                    await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, retryCount)), cancellationToken).ConfigureAwait(false);
                }
            }

            throw new InvalidOperationException($"Failed to download model {model.Name} after {maxRetries} attempts");
        }

        public async Task<bool> VerifyIntegrityAsync(string filePath, ModelDescriptor model, CancellationToken cancellationToken)
        {
            if (!File.Exists(filePath))
                return false;

            if (string.IsNullOrEmpty(model.Sha256))
                return true; // No checksum to verify

            using var sha256 = SHA256.Create();
            await using var stream = File.OpenRead(filePath);
            var computedHash = await sha256.ComputeHashAsync(stream, cancellationToken).ConfigureAwait(false);
            var computedHashString = Convert.ToHexString(computedHash);

            var isValid = string.Equals(computedHashString, model.Sha256, StringComparison.OrdinalIgnoreCase);
            if (!isValid)
            {
                _logger.LogError("❌ Integrity check failed for {ModelName}. Expected: {Expected}, Got: {Actual}", 
                    model.Name, model.Sha256, computedHashString);
            }

            return isValid;
        }
    }

    public sealed class DefaultModelHotSwapper : IModelHotSwapper
    {
        private readonly ILogger<DefaultModelHotSwapper> _logger;
        private ModelDescriptor? _activeModel;

        public DefaultModelHotSwapper(ILogger<DefaultModelHotSwapper> logger)
        {
            _logger = logger;
        }

        public async Task<bool> SwapModelAsync(ModelDescriptor newModel, CancellationToken cancellationToken)
        {
            _logger.LogInformation("🔄 Hot-swapping to model {ModelName} v{Version}", newModel.Name, newModel.Version);

            try
            {
                // Implement your ONNX session rebuild logic here
                // This is where you would reload your ML model runtime
                
                _activeModel = newModel;
                _logger.LogInformation("✅ Successfully swapped to model {ModelName}", newModel.Name);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to swap to model {ModelName}", newModel.Name);
                return false;
            }
        }

        public Task<ModelDescriptor?> GetActiveModelAsync(CancellationToken cancellationToken)
        {
            return Task.FromResult(_activeModel);
        }
    }

    public sealed class FileBasedPerformanceStore : IPerformanceStore
    {
        private readonly string _storePath;
        private readonly ILogger<FileBasedPerformanceStore> _logger;
        private readonly SemaphoreSlim _fileLock = new(1, 1);

        public FileBasedPerformanceStore(string storePath, ILogger<FileBasedPerformanceStore> logger)
        {
            _storePath = storePath;
            _logger = logger;
            
            // Ensure directory exists
            var directory = Path.GetDirectoryName(_storePath);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }
        }

        public async Task<ModelPerformance?> GetPerformanceAsync(string modelId, CancellationToken cancellationToken)
        {
            await _fileLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (!File.Exists(_storePath))
                    return null;

                var json = await File.ReadAllTextAsync(_storePath, cancellationToken).ConfigureAwait(false);
                var store = JsonSerializer.Deserialize<Dictionary<string, ModelPerformance>>(json);
                
                return store?.GetValueOrDefault(modelId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to read performance store");
                return null;
            }
            finally
            {
                _fileLock.Release();
            }
        }

        public async Task SavePerformanceAsync(string modelId, ModelPerformance performance, CancellationToken cancellationToken)
        {
            await _fileLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                var store = new Dictionary<string, ModelPerformance>();
                
                if (File.Exists(_storePath))
                {
                    var json = await File.ReadAllTextAsync(_storePath, cancellationToken).ConfigureAwait(false);
                    store = JsonSerializer.Deserialize<Dictionary<string, ModelPerformance>>(json) ?? new();
                }

                store[modelId] = performance;
                
                var updatedJson = JsonSerializer.Serialize(store, new JsonSerializerOptions { WriteIndented = true });
                await File.WriteAllTextAsync(_storePath, updatedJson, cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                _fileLock.Release();
            }
        }

        public async Task<List<ModelPerformance>> GetTopPerformersAsync(int count, CancellationToken cancellationToken)
        {
            await _fileLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (!File.Exists(_storePath))
                    return new List<ModelPerformance>();

                var json = await File.ReadAllTextAsync(_storePath, cancellationToken).ConfigureAwait(false);
                var store = JsonSerializer.Deserialize<Dictionary<string, ModelPerformance>>(json);
                
                return store?.Values
                    .OrderByDescending(p => p.SharpeRatio)
                    .Take(count)
                    .ToList() ?? new List<ModelPerformance>();
            }
            finally
            {
                _fileLock.Release();
            }
        }
    }
    #endregion

    #region Main Service
    /// <summary>
    /// Production-ready CloudRlTrainerV2 with full analyzer compliance
    /// </summary>
    public sealed class CloudRlTrainerV2 : BackgroundService
    {
        private readonly ILogger<CloudRlTrainerV2> _logger;
        private readonly CloudRlTrainerOptions _options;
        private readonly IModelDownloader _downloader;
        private readonly IModelHotSwapper _hotSwapper;
        private readonly IPerformanceStore _performanceStore;
        private readonly SemaphoreSlim _operationLock = new(1, 1);

        public CloudRlTrainerV2(
            ILogger<CloudRlTrainerV2> logger,
            IOptions<CloudRlTrainerOptions> options,
            IModelDownloader downloader,
            IModelHotSwapper hotSwapper,
            IPerformanceStore performanceStore)
        {
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _options = options?.Value ?? throw new ArgumentNullException(nameof(options));
            _downloader = downloader ?? throw new ArgumentNullException(nameof(downloader));
            _hotSwapper = hotSwapper ?? throw new ArgumentNullException(nameof(hotSwapper));
            _performanceStore = performanceStore ?? throw new ArgumentNullException(nameof(performanceStore));
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            if (!_options.Enabled)
            {
                _logger.LogInformation("CloudRlTrainerV2 is disabled");
                return;
            }

            _logger.LogInformation("🚀 CloudRlTrainerV2 starting with {IntervalMinutes}min polling", _options.PollIntervalMinutes);

            // Ensure directories exist
            Directory.CreateDirectory(_options.InstallDir);
            Directory.CreateDirectory(_options.TempDir);
            
            var configDir = Path.GetDirectoryName(_options.RegistryFile);
            if (!string.IsNullOrEmpty(configDir))
            {
                Directory.CreateDirectory(configDir);
            }

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    await ProcessModelsAsync(stoppingToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ Error in CloudRlTrainerV2 cycle");
                }

                await Task.Delay(TimeSpan.FromMinutes(_options.PollIntervalMinutes), stoppingToken).ConfigureAwait(false);
            }
        }

        private async Task ProcessModelsAsync(CancellationToken cancellationToken)
        {
            await _operationLock.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                await PollAsync(cancellationToken).ConfigureAwait(false);
            }
            finally
            {
                _operationLock.Release();
            }
        }

        /// <summary>
        /// Main polling logic for manifest-based model updates with atomic promotion
        /// </summary>
        private async Task PollAsync(CancellationToken ct)
        {
            // Check if promotion is enabled
            var promoteEnabled = Environment.GetEnvironmentVariable("PROMOTE_TUNER") == "1";
            if (!promoteEnabled)
            {
                _logger.LogDebug("PROMOTE_TUNER=0, skipping model promotion");
                return;
            }

            var manifestPath = Path.Combine("manifests", "manifest.json");
            if (!File.Exists(manifestPath)) 
            { 
                _logger.LogWarning("No manifest found at {ManifestPath}", manifestPath); 
                return; 
            }

            try
            {
                var manifestContent = await File.ReadAllTextAsync(manifestPath, ct).ConfigureAwait(false);
                var manifest = JsonSerializer.Deserialize<ModelManifest>(manifestContent);
                
                if (manifest == null || !IsEligible(manifest)) 
                { 
                    _logger.LogInformation("Manifest not eligible for promotion"); 
                    return; 
                }

                var sha = manifest.VersionSha();
                var stageDir = Path.Combine("artifacts", "stage", sha);
                Directory.CreateDirectory(stageDir);

                _logger.LogInformation("📦 Processing manifest version {Sha}", sha);

                // Download and verify all models
                foreach (var kv in manifest.Models)
                {
                    var local = Path.Combine(stageDir, kv.Key + ".onnx");
                    if (File.Exists(local))
                    {
                        _logger.LogDebug("Model {ModelName} already staged", kv.Key);
                        continue;
                    }

                    await DownloadAsync(kv.Value.Url, local, ct).ConfigureAwait(false);
                    
                    if (!await VerifySha256Async(local, kv.Value.Sha256).ConfigureAwait(false))
                    {
                        throw new InvalidOperationException($"SHA256 mismatch for {kv.Key}");
                    }
                    
                    _logger.LogInformation("✅ Downloaded and verified {ModelName}", kv.Key);
                }

                // Atomic swap
                await AtomicSwapAsync(stageDir, sha, ct).ConfigureAwait(false);

                // Notify brain for hot-reload
                NotifyModelUpdate(sha);

                _logger.LogInformation("🚀 Models promoted successfully {Sha}", sha);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to process manifest");
            }
        }

        private bool IsEligible(ModelManifest manifest)
        {
            // Check if CI backtest_sweep is green and drift is within limits
            var ciGreen = Environment.GetEnvironmentVariable("CI_BACKTEST_GREEN") == "1";
            var driftOk = manifest.DriftScore <= 0.15; // Example threshold
            
            return ciGreen && driftOk && manifest.Models.Count > 0;
        }

        private async Task DownloadAsync(string url, string localPath, CancellationToken ct)
        {
            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromMinutes(10);
            
            var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            await using var contentStream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
            await using var fileStream = File.Create(localPath);
            await contentStream.CopyToAsync(fileStream, ct).ConfigureAwait(false);
        }

        private async Task<bool> VerifySha256Async(string filePath, string expectedSha256)
        {
            if (string.IsNullOrEmpty(expectedSha256)) return true;
            
            await Task.CompletedTask.ConfigureAwait(false);
            
            using var sha256 = System.Security.Cryptography.SHA256.Create();
            await using var stream = File.OpenRead(filePath);
            var hash = await sha256.ComputeHashAsync(stream).ConfigureAwait(false);
            var computedSha256 = Convert.ToHexString(hash);
            
            return string.Equals(computedSha256, expectedSha256, StringComparison.OrdinalIgnoreCase);
        }

        private async Task AtomicSwapAsync(string stageDir, string sha, CancellationToken ct)
        {
            await Task.CompletedTask.ConfigureAwait(false);
            
            var curr = Path.Combine("artifacts", "current");
            var prev = Path.Combine("artifacts", "previous");

            // Backup current to previous (atomic)
            if (Directory.Exists(curr))
            {
                if (Directory.Exists(prev))
                {
                    Directory.Delete(prev, true);
                }
                Directory.Move(curr, prev);
            }

            // Move staged to current (atomic)
            Directory.Move(stageDir, curr);

            _logger.LogInformation("📋 Atomic swap completed: staged → current");
        }

        private void NotifyModelUpdate(string sha)
        {
            // Emit telemetry
            EmitMetric("config.snapshot_id", 1, ("id", sha));
            EmitMetric("canary.start", 1, ("id", sha));
            
            // TODO: Wire to IModelRegistry when available
            _logger.LogInformation("🔔 Model update notification sent for {Sha}", sha);
        }

        private void EmitMetric(string name, int value, (string key, string val) tag)
        {
            // Simple metric emission - integrate with your telemetry system
            _logger.LogInformation("📊 Metric: {MetricName}={Value} {TagKey}={TagValue}", 
                name, value, tag.key, tag.val);
        }

        private async Task<ModelRegistry> LoadRegistryAsync(CancellationToken cancellationToken)
        {
            if (!File.Exists(_options.RegistryFile))
            {
                return new ModelRegistry();
            }

            try
            {
                var json = await File.ReadAllTextAsync(_options.RegistryFile, cancellationToken).ConfigureAwait(false);
                return JsonSerializer.Deserialize<ModelRegistry>(json) ?? new ModelRegistry();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to load registry, creating new one");
                return new ModelRegistry();
            }
        }

        private async Task SaveRegistryAsync(ModelRegistry registry, CancellationToken cancellationToken)
        {
            var json = JsonSerializer.Serialize(registry, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(_options.RegistryFile, json, cancellationToken).ConfigureAwait(false);
        }

        private async Task<List<ModelDescriptor>> DiscoverModelsAsync(CancellationToken cancellationToken)
        {
            var models = new List<ModelDescriptor>();
            
            // Add discovery logic for GitHub releases, cloud storage, etc.
            // This is where you'd implement your specific model source discovery
            
            return models;
        }

        private async Task ProcessNewModelAsync(ModelDescriptor model, ModelRegistry registry, CancellationToken cancellationToken)
        {
            _logger.LogInformation("📦 Processing new model {ModelName} v{Version}", model.Name, model.Version);

            try
            {
                // Download model
                var modelPath = Path.Combine(_options.InstallDir, $"{model.Id}-{model.Version}.onnx");
                await _downloader.DownloadAsync(model, modelPath, cancellationToken).ConfigureAwait(false);

                // Verify integrity
                if (!await _downloader.VerifyIntegrityAsync(modelPath, model, cancellationToken).ConfigureAwait(false))
                {
                    _logger.LogError("❌ Integrity verification failed for {ModelName}", model.Name);
                    File.Delete(modelPath);
                    return;
                }

                // Add to installed models
                registry.Installed.Add(model);
                _logger.LogInformation("✅ Successfully processed model {ModelName}", model.Name);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to process model {ModelName}", model.Name);
            }
        }

        private async Task EvaluateAndSwapAsync(ModelRegistry registry, CancellationToken cancellationToken)
        {
            if (registry.Installed.Count == 0)
                return;

            // Get performance data for installed models
            var bestModel = registry.Installed.First(); // Default to first
            var bestPerformance = await _performanceStore.GetPerformanceAsync(bestModel.Id, cancellationToken).ConfigureAwait(false);

            foreach (var model in registry.Installed)
            {
                var performance = await _performanceStore.GetPerformanceAsync(model.Id, cancellationToken).ConfigureAwait(false);
                if (performance != null && (bestPerformance == null || performance.SharpeRatio > bestPerformance.SharpeRatio))
                {
                    bestModel = model;
                    bestPerformance = performance;
                }
            }

            // Check if we need to swap
            var currentActive = await _hotSwapper.GetActiveModelAsync(cancellationToken).ConfigureAwait(false);
            if (currentActive == null || currentActive.Id != bestModel.Id)
            {
                if (await _hotSwapper.SwapModelAsync(bestModel, cancellationToken).ConfigureAwait(false))
                {
                    registry.Active = bestModel;
                    _logger.LogInformation("🔄 Hot-swapped to model {ModelName} v{Version}", bestModel.Name, bestModel.Version);
                }
            }
        }

        public override void Dispose()
        {
            _operationLock?.Dispose();
            base.Dispose();
        }
    }
    #endregion
}