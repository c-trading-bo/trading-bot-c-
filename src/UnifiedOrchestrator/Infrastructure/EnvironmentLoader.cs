using System;
using System.IO;

namespace TradingBot.UnifiedOrchestrator.Infrastructure;

/// <summary>
/// Loads environment variables from .env files for unified configuration
/// </summary>
public static class EnvironmentLoader
{
    /// <summary>
    /// Load environment files in priority order: .env.local then .env
    /// Search current directory and up to 4 parent directories
    /// </summary>
    public static void LoadEnvironmentFiles()
    {
        try
        {
            // Search current and up to 4 parent directories for .env.local then .env
            var candidates = new[] { ".env.local", ".env" };
            string? dir = Environment.CurrentDirectory;
            
            for (int up; up < 5 && dir != null; up++)
            {
                foreach (var file in candidates)
                {
                    var path = Path.Combine(dir, file);
                    if (File.Exists(path))
                    {
                        LoadEnvironmentFile(path);
                        Console.WriteLine($"[EnvironmentLoader] Loaded {file} from {dir}");
                    }
                }
                dir = Directory.GetParent(dir)?.FullName;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[EnvironmentLoader] Warning: Failed to load .env files: {ex.Message}");
        }
    }
    
    /// <summary>
    /// Load a specific .env file and set environment variables
    /// </summary>
    private static void LoadEnvironmentFile(string filePath)
    {
        try
        {
            foreach (var raw in File.ReadAllLines(filePath))
            {
                var line = raw.Trim();
                if (line.Length == 0 || line.StartsWith('#')) continue;
                
                var idx = line.IndexOf('=');
                if (idx <= 0 || idx >= line.Length - 1) continue;
                
                var key = line.Substring(0, idx).Trim();
                var val = line.Substring(idx + 1).Trim();
                
                // Remove quotes if present
                if ((val.StartsWith('"') && val.EndsWith('"')) || 
                    (val.StartsWith('\'') && val.EndsWith('\'')))
                {
                    // Defensive bounds checking for quote removal
                    if (val.Length >= 2)
                    {
                        val = val.Substring(1, val.Length - 2);
                    }
                }
                
                if (!string.IsNullOrWhiteSpace(key))
                {
                    Environment.SetEnvironmentVariable(key, val);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[EnvironmentLoader] Warning: Failed to parse {filePath}: {ex.Message}");
        }
    }
}