using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TorchSharp;
using TorchSharp.Modules;

namespace TradingBot.RLAgent;

/// <summary>
/// Helper utilities for saving and validating TorchSharp models
/// Provides robust model persistence with validation
/// </summary>
public static class TorchModelSaveHelper
{
    /// <summary>
    /// Save a TorchSharp module with validation
    /// </summary>
    /// <param name="module">Module to save</param>
    /// <param name="path">Path to save to</param>
    /// <param name="modelName">Name of model for logging</param>
    /// <param name="minExpectedSizeBytes">Minimum expected file size in bytes (default 1KB)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    public static Task SaveModuleWithValidationAsync(
        torch.nn.Module module,
        string path,
        string modelName,
        long minExpectedSizeBytes = 1024,
        CancellationToken cancellationToken = default)
    {
        if (module == null)
        {
            throw new ArgumentNullException(nameof(module), $"{modelName}: Module is null, cannot save");
        }

        if (string.IsNullOrWhiteSpace(path))
        {
            throw new ArgumentNullException(nameof(path), $"{modelName}: Path is null or empty");
        }

        try
        {
            // Ensure parent directory exists
            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            // Save the module
            module.save(path);

            // Validate the saved file
            if (!File.Exists(path))
            {
                throw new IOException($"{modelName}: Model file was not created at path: {path}");
            }

            var fileInfo = new FileInfo(path);
            if (fileInfo.Length == 0)
            {
                throw new IOException($"{modelName}: Model file is empty (0 bytes): {path}");
            }

            if (fileInfo.Length < minExpectedSizeBytes)
            {
                throw new IOException(
                    $"{modelName}: Model file is suspiciously small ({fileInfo.Length} bytes, expected >= {minExpectedSizeBytes}): {path}. " +
                    "This may indicate a save failure or uninitialized weights.");
            }

            // Success - file saved and validated
            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            throw new IOException($"{modelName}: Failed to save model to {path}. Error: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Load a TorchSharp module with validation
    /// </summary>
    /// <param name="module">Module to load into</param>
    /// <param name="path">Path to load from</param>
    /// <param name="modelName">Name of model for logging</param>
    /// <param name="cancellationToken">Cancellation token</param>
    public static Task LoadModuleWithValidationAsync(
        torch.nn.Module module,
        string path,
        string modelName,
        CancellationToken cancellationToken = default)
    {
        if (module == null)
        {
            throw new ArgumentNullException(nameof(module), $"{modelName}: Module is null, cannot load");
        }

        if (!File.Exists(path))
        {
            throw new FileNotFoundException($"{modelName}: Model file not found at path: {path}", path);
        }

        var fileInfo = new FileInfo(path);
        if (fileInfo.Length == 0)
        {
            throw new IOException($"{modelName}: Model file is empty (0 bytes): {path}");
        }

        try
        {
            module.load(path);
            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            throw new IOException($"{modelName}: Failed to load model from {path}. Error: {ex.Message}", ex);
        }
    }
}
