using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Platform-specific Python path detection and configuration
/// Fixes Python path issues on Windows vs Linux environments
/// </summary>
public interface IPythonPathResolver
{
    string GetPythonExecutablePath();
    string GetWorkingDirectory();
    string ResolvePythonScriptPath(string relativePath);
}

public class PlatformAwarePythonPathResolver : IPythonPathResolver
{
    private readonly ILogger<PlatformAwarePythonPathResolver> _logger;
    private readonly ITradingLogger _tradingLogger;
    private string? _cachedPythonPath;

    public PlatformAwarePythonPathResolver(
        ILogger<PlatformAwarePythonPathResolver> logger,
        ITradingLogger tradingLogger)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
    }

    public string GetPythonExecutablePath()
    {
        if (!string.IsNullOrEmpty(_cachedPythonPath))
        {
            return _cachedPythonPath;
        }

        // Check environment variable first
        var envPython = Environment.GetEnvironmentVariable("PYTHON_EXECUTABLE");
        if (!string.IsNullOrEmpty(envPython) && File.Exists(envPython))
        {
            _cachedPythonPath = envPython;
            _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "PythonResolver", 
                $"Using Python from environment: {_cachedPythonPath}");
            return _cachedPythonPath;
        }

        // Platform-specific detection
        var candidates = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? GetWindowsPythonCandidates()
            : GetLinuxPythonCandidates();

        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                // Verify it's actually Python
                if (IsPythonExecutable(candidate))
                {
                    _cachedPythonPath = candidate;
                    _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "PythonResolver", 
                        $"Detected Python executable: {_cachedPythonPath}");
                    return _cachedPythonPath;
                }
            }
        }

        // Fallback
        _cachedPythonPath = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "python.exe" : "python3";
        _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "PythonResolver", 
            $"Using fallback Python path: {_cachedPythonPath}");
        
        return _cachedPythonPath;
    }

    public string GetWorkingDirectory()
    {
        // Use repository root as working directory
        var currentDir = Directory.GetCurrentDirectory();
        
        // Find the repository root (contains .git or specific files)
        var dir = currentDir;
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir, ".git")) || 
                File.Exists(Path.Combine(dir, "TopstepX.Bot.sln")))
            {
                return dir;
            }
            dir = Directory.GetParent(dir)?.FullName;
        }

        return currentDir;
    }

    public string ResolvePythonScriptPath(string relativePath)
    {
        var workingDir = GetWorkingDirectory();
        var scriptPath = Path.Combine(workingDir, relativePath.TrimStart('.', '/', '\\'));
        
        // Convert to platform-specific path
        return Path.GetFullPath(scriptPath);
    }

    private static string[] GetWindowsPythonCandidates()
    {
        var userProfile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        var programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
        var programFilesX86 = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86);
        
        return new[]
        {
            "python.exe",
            "python3.exe",
            Path.Combine(userProfile, "AppData", "Local", "Programs", "Python", "Python311", "python.exe"),
            Path.Combine(userProfile, "AppData", "Local", "Programs", "Python", "Python310", "python.exe"),
            Path.Combine(userProfile, "AppData", "Local", "Programs", "Python", "Python39", "python.exe"),
            Path.Combine(programFiles, "Python311", "python.exe"),
            Path.Combine(programFiles, "Python310", "python.exe"),
            Path.Combine(programFilesX86, "Python311", "python.exe"),
            Path.Combine(programFilesX86, "Python310", "python.exe"),
            @"C:\Python311\python.exe",
            @"C:\Python310\python.exe",
            @"C:\Python39\python.exe"
        };
    }

    private static string[] GetLinuxPythonCandidates()
    {
        return new[]
        {
            "/usr/bin/python3",
            "/usr/bin/python",
            "/usr/local/bin/python3",
            "/usr/local/bin/python",
            "/opt/python3/bin/python3",
            "python3",
            "python"
        };
    }

    private bool IsPythonExecutable(string path)
    {
        try
        {
            var startInfo = new ProcessStartInfo
            {
                FileName = path,
                Arguments = "--version",
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            using var process = Process.Start(startInfo);
            if (process == null) return false;

            process.WaitForExit(5000); // 5 second timeout
            var output = process.StandardOutput.ReadToEnd();
            
            return output.Contains("Python") && process.ExitCode == 0;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to verify Python executable: {Path}", path);
            return false;
        }
    }
}