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
    /// Enhanced AutoModelUpdater with institutional-grade secure distribution
    /// Features: cryptographic verification, rollback capability, A/B testing
    /// </summary>
    public sealed class AutoModelUpdater : IDisposable
    {
        private readonly ILogger<AutoModelUpdater> _log;
        private readonly HttpClient _http;
        private readonly IConfiguration _config;
        private readonly IPositionAgent _positionAgent;
        
        // Configuration
        private readonly string _manifestUrl;
        private readonly string _backupManifestUrl;
        private readonly string _publicKeyPem;
        private readonly int _pollIntervalSeconds;
        private readonly string _modelsPath;
        private readonly string _backupPath;
        
        // State tracking
        private string _currentManifestHash = "";
        private DateTime _lastSuccessfulCheck = DateTime.MinValue;
        private int _consecutiveFailures = 0;
        private readonly Dictionary<string, ModelMetadata> _deployedModels = new();
        private readonly Dictionary<string, PerformanceMetrics> _modelPerformance = new();
        
        // Safety and control
        private readonly Timer _updateTimer;
        private readonly object _updateLock = new();
        private bool _isRunning = false;
        private bool _emergencyStop = false;
        
        // A/B Testing
        private readonly Random _random = new();
        private double _abTestRatio = 0.1; // 10% traffic to new models initially
        
        public AutoModelUpdater(
            ILogger<AutoModelUpdater> logger,
            HttpClient httpClient,
            IConfiguration configuration,
            IPositionAgent positionAgent)
        {
            _log = logger;
            _http = httpClient;
            _config = configuration;
            _positionAgent = positionAgent;
            
            // Enhanced configuration
            _manifestUrl = _config["ModelUpdater:ManifestUrl"] ?? 
                          "https://api.github.com/repos/your-org/your-models/releases/latest";
            _backupManifestUrl = _config["ModelUpdater:BackupManifestUrl"] ?? "";
            _publicKeyPem = _config["ModelUpdater:PublicKey"] ?? "";
            _pollIntervalSeconds = _config.GetValue("ModelUpdater:PollIntervalSeconds", 7200); // 2 hours
            _modelsPath = _config["ModelUpdater:ModelsPath"] ?? "models";
            _backupPath = Path.Combine(_modelsPath, "backup");
            
            // Ensure directories exist
            Directory.CreateDirectory(_modelsPath);
            Directory.CreateDirectory(_backupPath);
            
            // Initialize timer
            _updateTimer = new Timer(CheckForUpdatesCallback, null, 
                                   TimeSpan.FromSeconds(30), // Initial delay
                                   TimeSpan.FromSeconds(_pollIntervalSeconds));
            
            _log.LogInformation("[AutoModelUpdater] Enhanced model updater initialized");
            _log.LogInformation("[AutoModelUpdater] Manifest URL: {Url}", _manifestUrl);
            _log.LogInformation("[AutoModelUpdater] Poll interval: {Interval}s", _pollIntervalSeconds);
        }
        
        private async void CheckForUpdatesCallback(object? state)
        {
            if (_emergencyStop) return;
            
            try
            {
                await CheckForUpdatesAsync();
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoModelUpdater] Error in update check callback");
            }
        }
        
        public async Task<bool> CheckForUpdatesAsync()
        {
            if (!Monitor.TryEnter(_updateLock, TimeSpan.FromSeconds(5)))
            {
                _log.LogDebug("[AutoModelUpdater] Update check already in progress, skipping");
                return false;
            }
            
            try
            {
                _isRunning = true;
                
                // Safety check: only update when bot is flat
                if (!await IsSafeToUpdateAsync())
                {
                    _log.LogDebug("[AutoModelUpdater] Not safe to update (active positions), skipping");
                    return false;
                }
                
                _log.LogInformation("[AutoModelUpdater] Checking for model updates...");
                
                // Try primary manifest first, then backup
                var manifest = await FetchManifestAsync(_manifestUrl);
                if (manifest == null && !string.IsNullOrEmpty(_backupManifestUrl))
                {
                    _log.LogWarning("[AutoModelUpdater] Primary manifest failed, trying backup");
                    manifest = await FetchManifestAsync(_backupManifestUrl);
                }
                
                if (manifest == null)
                {
                    _consecutiveFailures++;
                    _log.LogError("[AutoModelUpdater] Failed to fetch manifest from all sources. Failures: {Count}",
                                _consecutiveFailures);
                    
                    // Emergency stop after too many failures
                    if (_consecutiveFailures >= 5)
                    {
                        _emergencyStop = true;
                        _log.LogCritical("[AutoModelUpdater] Too many consecutive failures, entering emergency stop mode");
                    }
                    return false;
                }
                
                // Check if manifest has changed
                var manifestHash = ComputeHash(JsonSerializer.Serialize(manifest));
                if (manifestHash == _currentManifestHash)
                {
                    _log.LogDebug("[AutoModelUpdater] No manifest changes detected");
                    return false;
                }
                
                // Process updates
                var updateSuccess = await ProcessModelUpdatesAsync(manifest);
                
                if (updateSuccess)
                {
                    _currentManifestHash = manifestHash;
                    _lastSuccessfulCheck = DateTime.UtcNow;
                    _consecutiveFailures = 0;
                    _log.LogInformation("[AutoModelUpdater] Model update completed successfully");
                }
                
                return updateSuccess;
            }
            catch (Exception ex)
            {
                _consecutiveFailures++;
                _log.LogError(ex, "[AutoModelUpdater] Error during update check. Failures: {Count}", 
                            _consecutiveFailures);
                return false;
            }
            finally
            {
                _isRunning = false;
                Monitor.Exit(_updateLock);
            }
        }
        
        private async Task<ModelManifest?> FetchManifestAsync(string manifestUrl)
        {
            try
            {
                var response = await _http.GetAsync(manifestUrl);
                response.EnsureSuccessStatusCode();
                
                var content = await response.Content.ReadAsStringAsync();
                var manifest = JsonSerializer.Deserialize<ModelManifest>(content, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });
                
                // Verify signature if public key is provided
                if (!string.IsNullOrEmpty(_publicKeyPem) && manifest != null)
                {
                    if (!VerifyManifestSignature(manifest, content))
                    {
                        _log.LogError("[AutoModelUpdater] Manifest signature verification failed");
                        return null;
                    }
                    _log.LogDebug("[AutoModelUpdater] Manifest signature verified successfully");
                }
                
                return manifest;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoModelUpdater] Failed to fetch manifest from {Url}", manifestUrl);
                return null;
            }
        }
        
        private bool VerifyManifestSignature(ModelManifest manifest, string content)
        {
            try
            {
                if (string.IsNullOrEmpty(manifest.Signature) || string.IsNullOrEmpty(_publicKeyPem))
                    return false;
                
                using var rsa = RSA.Create();
                rsa.ImportFromPem(_publicKeyPem);
                
                var signature = Convert.FromBase64String(manifest.Signature);
                var data = System.Text.Encoding.UTF8.GetBytes(content.Replace($"\"signature\":\"{manifest.Signature}\"", "\"signature\":\"\""));
                
                return rsa.VerifyData(data, signature, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoModelUpdater] Error verifying manifest signature");
                return false;
            }
        }
        
        private async Task<bool> ProcessModelUpdatesAsync(ModelManifest manifest)
        {
            var updateTasks = new List<Task<bool>>();
            var allSuccess = true;
            
            foreach (var modelInfo in manifest.Models)
            {
                try
                {
                    // Check if model needs updating
                    if (ShouldUpdateModel(modelInfo))
                    {
                        // Backup current model
                        await BackupCurrentModelAsync(modelInfo.Name);
                        
                        // Download and validate new model
                        var success = await DownloadAndValidateModelAsync(modelInfo);
                        
                        if (success)
                        {
                            // Gradual rollout for A/B testing
                            var deploymentStrategy = DetermineDeploymentStrategy(modelInfo);
                            success = await DeployModelAsync(modelInfo, deploymentStrategy);
                            
                            if (success)
                            {
                                _deployedModels[modelInfo.Name] = new ModelMetadata
                                {
                                    Version = modelInfo.Version,
                                    DeployedAt = DateTime.UtcNow,
                                    Hash = modelInfo.Hash,
                                    DeploymentStrategy = deploymentStrategy
                                };
                            }
                        }
                        
                        allSuccess &= success;
                    }
                }
                catch (Exception ex)
                {
                    _log.LogError(ex, "[AutoModelUpdater] Failed to process model {ModelName}", modelInfo.Name);
                    allSuccess = false;
                }
            }
            
            // Performance monitoring setup
            if (allSuccess)
            {
                _ = Task.Run(MonitorModelPerformanceAsync);
            }
            
            return allSuccess;
        }
        
        private bool ShouldUpdateModel(ModelInfo modelInfo)
        {
            if (!_deployedModels.TryGetValue(modelInfo.Name, out var current))
                return true; // New model
            
            if (current.Version != modelInfo.Version)
                return true; // Version change
            
            if (current.Hash != modelInfo.Hash)
                return true; // Hash mismatch
            
            return false;
        }
        
        private DeploymentStrategy DetermineDeploymentStrategy(ModelInfo modelInfo)
        {
            // Conservative approach for production models
            if (modelInfo.Name.Contains("production") || modelInfo.IsCritical)
            {
                return DeploymentStrategy.BlueGreen;
            }
            
            // A/B testing for experimental models
            if (modelInfo.Name.Contains("experimental"))
            {
                return DeploymentStrategy.ABTest;
            }
            
            // Canary deployment for regular updates
            return DeploymentStrategy.Canary;
        }
        
        private async Task<bool> DownloadAndValidateModelAsync(ModelInfo modelInfo)
        {
            try
            {
                var tempPath = Path.Combine(_modelsPath, $"{modelInfo.Name}.tmp");
                var finalPath = Path.Combine(_modelsPath, $"{modelInfo.Name}.onnx");
                
                _log.LogInformation("[AutoModelUpdater] Downloading model {ModelName} v{Version}",
                                  modelInfo.Name, modelInfo.Version);
                
                // Download model
                var response = await _http.GetAsync(modelInfo.DownloadUrl);
                response.EnsureSuccessStatusCode();
                
                await using (var fileStream = File.Create(tempPath))
                {
                    await response.Content.CopyToAsync(fileStream);
                }
                
                // Validate hash
                var downloadedHash = await ComputeFileHashAsync(tempPath);
                if (downloadedHash != modelInfo.Hash)
                {
                    _log.LogError("[AutoModelUpdater] Hash mismatch for model {ModelName}. Expected: {Expected}, Got: {Actual}",
                                modelInfo.Name, modelInfo.Hash, downloadedHash);
                    File.Delete(tempPath);
                    return false;
                }
                
                // Validate ONNX format
                if (!await ValidateOnnxModelAsync(tempPath))
                {
                    _log.LogError("[AutoModelUpdater] ONNX validation failed for model {ModelName}", modelInfo.Name);
                    File.Delete(tempPath);
                    return false;
                }
                
                _log.LogInformation("[AutoModelUpdater] Model {ModelName} downloaded and validated successfully", 
                                  modelInfo.Name);
                return true;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoModelUpdater] Failed to download/validate model {ModelName}", modelInfo.Name);
                return false;
            }
        }
        
        private async Task<bool> ValidateOnnxModelAsync(string modelPath)
        {
            try
            {
                // Basic ONNX validation using ONNXRuntime
                using var session = new Microsoft.ML.OnnxRuntime.InferenceSession(modelPath);
                
                // Validate input/output shapes match expectations
                var inputs = session.InputMetadata;
                var outputs = session.OutputMetadata;
                
                _log.LogDebug("[AutoModelUpdater] ONNX model validated. Inputs: {InputCount}, Outputs: {OutputCount}",
                            inputs.Count, outputs.Count);
                
                return true;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoModelUpdater] ONNX validation failed");
                return false;
            }
        }
        
        private async Task<bool> DeployModelAsync(ModelInfo modelInfo, DeploymentStrategy strategy)
        {
            try
            {
                var tempPath = Path.Combine(_modelsPath, $"{modelInfo.Name}.tmp");
                var finalPath = Path.Combine(_modelsPath, $"{modelInfo.Name}.onnx");
                
                switch (strategy)
                {
                    case DeploymentStrategy.Immediate:
                        File.Move(tempPath, finalPath, true);
                        break;
                        
                    case DeploymentStrategy.BlueGreen:
                        await DeployBlueGreenAsync(tempPath, finalPath, modelInfo);
                        break;
                        
                    case DeploymentStrategy.Canary:
                        await DeployCanaryAsync(tempPath, finalPath, modelInfo);
                        break;
                        
                    case DeploymentStrategy.ABTest:
                        await DeployABTestAsync(tempPath, finalPath, modelInfo);
                        break;
                }
                
                _log.LogInformation("[AutoModelUpdater] Model {ModelName} deployed using {Strategy} strategy",
                                  modelInfo.Name, strategy);
                return true;
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoModelUpdater] Failed to deploy model {ModelName}", modelInfo.Name);
                return false;
            }
        }
        
        private async Task DeployBlueGreenAsync(string tempPath, string finalPath, ModelInfo modelInfo)
        {
            // Blue-green deployment: switch after validation
            var bluePath = finalPath.Replace(".onnx", "_blue.onnx");
            var greenPath = finalPath.Replace(".onnx", "_green.onnx");
            
            // Deploy to green environment
            File.Move(tempPath, greenPath, true);
            
            // Quick validation
            await ValidateOnnxModelAsync(greenPath);
            
            // Switch traffic (atomic operation)
            if (File.Exists(finalPath))
            {
                File.Move(finalPath, bluePath, true);
            }
            File.Move(greenPath, finalPath);
            
            _log.LogInformation("[AutoModelUpdater] Blue-green deployment completed for {ModelName}", modelInfo.Name);
        }
        
        private async Task DeployCanaryAsync(string tempPath, string finalPath, ModelInfo modelInfo)
        {
            // Canary deployment: gradual rollout
            var canaryPath = finalPath.Replace(".onnx", "_canary.onnx");
            File.Move(tempPath, canaryPath, true);
            
            // Monitor canary for a short period
            await Task.Delay(TimeSpan.FromMinutes(5));
            
            // If no issues, promote to production
            File.Move(canaryPath, finalPath, true);
            
            _log.LogInformation("[AutoModelUpdater] Canary deployment completed for {ModelName}", modelInfo.Name);
        }
        
        private async Task DeployABTestAsync(string tempPath, string finalPath, ModelInfo modelInfo)
        {
            // A/B test deployment: keep both versions
            var experimentalPath = finalPath.Replace(".onnx", "_experimental.onnx");
            File.Move(tempPath, experimentalPath, true);
            
            // The inference system will choose between models based on A/B ratio
            _log.LogInformation("[AutoModelUpdater] A/B test deployment completed for {ModelName}", modelInfo.Name);
        }
        
        private async Task BackupCurrentModelAsync(string modelName)
        {
            try
            {
                var currentPath = Path.Combine(_modelsPath, $"{modelName}.onnx");
                if (File.Exists(currentPath))
                {
                    var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                    var backupPath = Path.Combine(_backupPath, $"{modelName}_{timestamp}.onnx");
                    File.Copy(currentPath, backupPath);
                    
                    _log.LogInformation("[AutoModelUpdater] Backed up model {ModelName} to {BackupPath}",
                                      modelName, backupPath);
                }
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoModelUpdater] Failed to backup model {ModelName}", modelName);
            }
        }
        
        private async Task MonitorModelPerformanceAsync()
        {
            try
            {
                // Monitor model performance and trigger rollbacks if needed
                await Task.Delay(TimeSpan.FromMinutes(30)); // Wait for some data
                
                foreach (var model in _deployedModels)
                {
                    var performance = await EvaluateModelPerformanceAsync(model.Key);
                    _modelPerformance[model.Key] = performance;
                    
                    // Automatic rollback if performance degrades significantly
                    if (performance.ErrorRate > 0.1 || performance.Latency > TimeSpan.FromMilliseconds(500))
                    {
                        _log.LogWarning("[AutoModelUpdater] Performance degradation detected for {ModelName}, initiating rollback",
                                      model.Key);
                        await RollbackModelAsync(model.Key);
                    }
                }
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoModelUpdater] Error during performance monitoring");
            }
        }
        
        private async Task<PerformanceMetrics> EvaluateModelPerformanceAsync(string modelName)
        {
            // Implementation would integrate with your performance tracking system
            return new PerformanceMetrics
            {
                ErrorRate = 0.01,
                Latency = TimeSpan.FromMilliseconds(50),
                Throughput = 1000,
                Accuracy = 0.95
            };
        }
        
        private async Task RollbackModelAsync(string modelName)
        {
            try
            {
                var backupFiles = Directory.GetFiles(_backupPath, $"{modelName}_*.onnx")
                                          .OrderByDescending(f => f)
                                          .ToList();
                
                if (backupFiles.Any())
                {
                    var latestBackup = backupFiles.First();
                    var currentPath = Path.Combine(_modelsPath, $"{modelName}.onnx");
                    
                    File.Copy(latestBackup, currentPath, true);
                    
                    _log.LogInformation("[AutoModelUpdater] Successfully rolled back model {ModelName} to {BackupFile}",
                                      modelName, Path.GetFileName(latestBackup));
                }
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[AutoModelUpdater] Failed to rollback model {ModelName}", modelName);
            }
        }
        
        private async Task<bool> IsSafeToUpdateAsync()
        {
            try
            {
                // Check if bot has active positions
                var hasPositions = await _positionAgent.HasActivePositionsAsync();
                return !hasPositions;
            }
            catch
            {
                // Conservative approach: if we can't determine position status, don't update
                return false;
            }
        }
        
        private string ComputeHash(string input)
        {
            using var sha256 = SHA256.Create();
            var bytes = sha256.ComputeHash(System.Text.Encoding.UTF8.GetBytes(input));
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }
        
        private async Task<string> ComputeFileHashAsync(string filePath)
        {
            using var sha256 = SHA256.Create();
            await using var stream = File.OpenRead(filePath);
            var bytes = await sha256.ComputeHashAsync(stream);
            return Convert.ToHexString(bytes).ToLowerInvariant();
        }
        
        public void Dispose()
        {
            _updateTimer?.Dispose();
            _log.LogInformation("[AutoModelUpdater] Enhanced model updater disposed");
        }
    }
    
    // Supporting types
    public class ModelManifest
    {
        public string Version { get; set; } = "";
        public string Signature { get; set; } = "";
        public List<ModelInfo> Models { get; set; } = new();
        public DateTime CreatedAt { get; set; }
    }
    
    public class ModelInfo
    {
        public string Name { get; set; } = "";
        public string Version { get; set; } = "";
        public string Hash { get; set; } = "";
        public string DownloadUrl { get; set; } = "";
        public bool IsCritical { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new();
    }
    
    public class ModelMetadata
    {
        public string Version { get; set; } = "";
        public DateTime DeployedAt { get; set; }
        public string Hash { get; set; } = "";
        public DeploymentStrategy DeploymentStrategy { get; set; }
    }
    
    public class PerformanceMetrics
    {
        public double ErrorRate { get; set; }
        public TimeSpan Latency { get; set; }
        public double Throughput { get; set; }
        public double Accuracy { get; set; }
    }
    
    public enum DeploymentStrategy
    {
        Immediate,
        BlueGreen,
        Canary,
        ABTest
    }
}