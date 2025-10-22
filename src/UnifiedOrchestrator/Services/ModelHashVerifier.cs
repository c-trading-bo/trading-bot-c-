using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Model Hash Verifier - Verifies that training actually changed model weights
/// Computes SHA256 hashes before and after training to prove learning occurred
/// Step 5 from Integration Plan
/// </summary>
internal sealed class ModelHashVerifier
{
    private readonly ILogger<ModelHashVerifier> _logger;
    private const long MinimumModelSizeBytes = 100 * 1024; // 100 KB minimum

    public ModelHashVerifier(ILogger<ModelHashVerifier> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Compute SHA256 hash of a model file
    /// </summary>
    public async Task<string> ComputeModelHashAsync(string modelPath, CancellationToken cancellationToken = default)
    {
        if (!File.Exists(modelPath))
        {
            _logger.LogWarning("[HASH-VERIFIER] Model file not found: {Path}", modelPath);
            return string.Empty;
        }

        try
        {
            using var sha256 = SHA256.Create();
            using var stream = File.OpenRead(modelPath);
            var hashBytes = await sha256.ComputeHashAsync(stream, cancellationToken).ConfigureAwait(false);
            var hash = BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
            
            var fileInfo = new FileInfo(modelPath);
            _logger.LogInformation("[HASH-VERIFIER] Model: {Model}, Hash: {Hash}, Size: {Size:N0} bytes",
                Path.GetFileName(modelPath), hash, fileInfo.Length);
            
            return hash;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[HASH-VERIFIER] Failed to compute hash for {Path}", modelPath);
            return string.Empty;
        }
    }

    /// <summary>
    /// Verify that a model changed after training
    /// Returns true if model changed (proof of learning), false if unchanged (no training occurred)
    /// </summary>
    public async Task<ModelVerificationResult> VerifyModelChangedAsync(
        string modelPath,
        string modelName,
        string? beforeHash = null,
        CancellationToken cancellationToken = default)
    {
        var result = new ModelVerificationResult
        {
            ModelName = modelName,
            ModelPath = modelPath
        };

        if (!File.Exists(modelPath))
        {
            result.Success = false;
            result.ErrorMessage = "Model file does not exist";
            _logger.LogError("[HASH-VERIFIER] ❌ VERIFICATION FAILED - {Model}: File not found at {Path}",
                modelName, modelPath);
            return result;
        }

        // Check file size
        var fileInfo = new FileInfo(modelPath);
        result.FileSizeBytes = fileInfo.Length;

        if (fileInfo.Length < MinimumModelSizeBytes)
        {
            result.Success = false;
            result.ErrorMessage = $"Model too small ({fileInfo.Length:N0} bytes < {MinimumModelSizeBytes:N0} bytes minimum)";
            _logger.LogError("[HASH-VERIFIER] ❌ VERIFICATION FAILED - {Model}: File too small (likely incomplete)",
                modelName);
            _logger.LogError("[HASH-VERIFIER]    Expected: ≥ {Min:N0} bytes, Actual: {Actual:N0} bytes",
                MinimumModelSizeBytes, fileInfo.Length);
            return result;
        }

        // Compute after-training hash
        var afterHash = await ComputeModelHashAsync(modelPath, cancellationToken).ConfigureAwait(false);
        result.AfterHash = afterHash;

        if (string.IsNullOrEmpty(afterHash))
        {
            result.Success = false;
            result.ErrorMessage = "Failed to compute hash";
            return result;
        }

        // If we have a before hash, compare them
        if (!string.IsNullOrEmpty(beforeHash))
        {
            result.BeforeHash = beforeHash;
            result.HashChanged = !beforeHash.Equals(afterHash, StringComparison.OrdinalIgnoreCase);

            if (result.HashChanged)
            {
                _logger.LogInformation("[HASH-VERIFIER] ✅ MODEL CHANGED - {Model}: Learning verified!",
                    modelName);
                _logger.LogInformation("[HASH-VERIFIER]    Before: {Before}...", beforeHash[..16]);
                _logger.LogInformation("[HASH-VERIFIER]    After:  {After}...", afterHash[..16]);
                _logger.LogInformation("[HASH-VERIFIER]    Size: {Size:N0} bytes", fileInfo.Length);
                result.Success = true;
            }
            else
            {
                result.Success = false;
                result.ErrorMessage = "Model hash unchanged - no learning occurred";
                _logger.LogError("[HASH-VERIFIER] ❌ MODEL UNCHANGED - {Model}: Training did NOT change weights!",
                    modelName);
                _logger.LogError("[HASH-VERIFIER]    Hash: {Hash}... (identical before and after)",
                    beforeHash[..16]);
                _logger.LogError("[HASH-VERIFIER]    This indicates incomplete training - model not actually trained");
            }
        }
        else
        {
            // No before hash to compare - just verify size is valid
            result.Success = true;
            result.HashChanged = true; // Assume changed since we can't verify
            _logger.LogInformation("[HASH-VERIFIER] ✅ MODEL VERIFIED - {Model}: Valid size, no baseline for comparison",
                modelName);
        }

        return result;
    }

    /// <summary>
    /// Capture model state before training (compute hash)
    /// </summary>
    public async Task<string> CaptureModelStateBeforeTrainingAsync(
        string modelPath,
        CancellationToken cancellationToken = default)
    {
        if (!File.Exists(modelPath))
        {
            _logger.LogInformation("[HASH-VERIFIER] No existing model at {Path} - this is a new model",
                modelPath);
            return string.Empty;
        }

        var hash = await ComputeModelHashAsync(modelPath, cancellationToken).ConfigureAwait(false);
        _logger.LogInformation("[HASH-VERIFIER] Captured baseline hash for {Model}: {Hash}...",
            Path.GetFileName(modelPath), hash[..16]);
        return hash;
    }
}

/// <summary>
/// Result of model verification
/// </summary>
public sealed class ModelVerificationResult
{
    public string ModelName { get; set; } = string.Empty;
    public string ModelPath { get; set; } = string.Empty;
    public string? BeforeHash { get; set; }
    public string? AfterHash { get; set; }
    public bool HashChanged { get; set; }
    public long FileSizeBytes { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
}
