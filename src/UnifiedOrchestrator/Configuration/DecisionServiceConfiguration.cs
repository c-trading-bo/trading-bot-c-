using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Net.Http;
using Microsoft.Extensions.Logging;
using System.IO;

namespace TradingBot.UnifiedOrchestrator.Configuration;

/// <summary>
/// Configuration for workflow scheduling system
/// </summary>
public class WorkflowSchedulingOptions
{
    public bool Enabled { get; set; } = true;
    public Dictionary<string, WorkflowScheduleConfig> DefaultSchedules { get; } = new();
    public List<string> MarketHolidays { get; } = new();
    public string TimeZone { get; set; } = "America/New_York";
}

/// <summary>
/// Configuration for individual workflow schedule
/// </summary>
public class WorkflowScheduleConfig
{
    public string? MarketHours { get; set; }
    public string? ExtendedHours { get; set; }
    public string? Overnight { get; set; }
    public string? CoreHours { get; set; }
    public string? FirstHour { get; set; }
    public string? LastHour { get; set; }
    public string? Regular { get; set; }
    public string? Global { get; set; }
    public string? Weekends { get; set; }
    public string? Disabled { get; set; }
    
    // CME Futures Session Configuration
    public string? SessionOpen { get; set; }
    public string? SessionClose { get; set; }
    public string? DailyBreakStart { get; set; }
    public string? DailyBreakEnd { get; set; }
}

/// <summary>
/// Configuration for Python integration
/// </summary>
public class PythonIntegrationOptions
{
    public bool Enabled { get; set; } = true;
    public string PythonPath { get; set; } = "/usr/bin/python3";
    public string WorkingDirectory { get; set; } = "./python";
    public Dictionary<string, string> ScriptPaths { get; } = new();
    public int Timeout { get; set; } = 30;
}

/// <summary>
/// Configuration for model loading system
/// </summary>
public class ModelLoadingOptions
{
    public bool Enabled { get; set; } = true;
    public bool OnnxEnabled { get; set; } = true;
    public string ModelsDirectory { get; set; } = "./models";
    public string FallbackMode { get; set; } = "simulation";
    public Dictionary<string, string> ModelPaths { get; } = new();
    public int HealthCheckInterval { get; set; } = 300;
}

/// <summary>
/// Configuration options for DecisionServiceLauncher
/// </summary>
public class DecisionServiceLauncherOptions
{
    public string ExecutablePath { get; set; } = "decision-service";
    public string WorkingDirectory { get; set; } = "./decision-service";
    public string ScriptPath { get; set; } = "./decision-service/start.py";
    public string ConfigFile { get; set; } = "./decision-service/config.json";
    public string PythonExecutable { get; set; } = "python";
    public int TimeoutSeconds { get; set; } = 300;
    public int StartupTimeoutSeconds { get; set; } = 60;
    public string Host { get; set; } = "localhost";
    public int Port { get; set; } = 8080;
    public bool EnableLogging { get; set; } = true;
    public bool Enabled { get; set; };
    public bool AutoRestart { get; set; } = true;
    public Dictionary<string, string> Environment { get; } = new();
}

/// <summary>
/// Configuration options for DecisionService
/// </summary>
public class DecisionServiceOptions
{
    public string ServiceUrl { get; set; } = "http://localhost:8080";
    public string BaseUrl { get; set; } = "http://localhost:8080";
    public string ApiKey { get; set; } = string.Empty;
    public int RequestTimeoutSeconds { get; set; } = 30;
    public int TimeoutMs { get; set; } = 30000;
    public int MaxRetries { get; set; } = 3;
    public bool EnableHealthChecks { get; set; } = true;
    public bool Enabled { get; set; };
}

/// <summary>
/// Configuration options for DecisionServiceIntegration
/// </summary>
public class DecisionServiceIntegrationOptions
{
    public string IntegrationEndpoint { get; set; } = "http://localhost:8080/integration";
    public string WebhookUrl { get; set; } = string.Empty;
    public int SyncIntervalSeconds { get; set; } = 60;
    public bool EnableRealTimeSync { get; set; };
    public bool Enabled { get; set; };
    public int HealthCheckIntervalSeconds { get; set; } = 30;
    public bool LogDecisionLines { get; set; } = true;
    public bool EnableTradeManagement { get; set; };
    public Dictionary<string, object> CustomSettings { get; } = new();
}

/// <summary>
/// Client for DecisionService with Python integration
/// </summary>
public class DecisionServiceClient
{
    private readonly DecisionServiceOptions _options;
    private readonly HttpClient _httpClient;
    private readonly PythonIntegrationOptions _pythonOptions;
    private readonly ILogger<DecisionServiceClient>? _logger;

    public DecisionServiceClient(DecisionServiceOptions options, HttpClient httpClient, PythonIntegrationOptions pythonOptions, ILogger<DecisionServiceClient>? logger = null)
    {
        _options = options;
        _httpClient = httpClient;
        _pythonOptions = pythonOptions;
        _logger = logger;
        _httpClient.BaseAddress = new Uri(_options.ServiceUrl);
        _httpClient.Timeout = TimeSpan.FromSeconds(_options.RequestTimeoutSeconds);
    }

    public async Task<bool> IsHealthyAsync(CancellationToken cancellationToken = default)
    {
        if (!_options.EnableHealthChecks) return true;
        
        try
        {
            var response = await _httpClient.GetAsync("/health", cancellationToken).ConfigureAwait(false);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task<string> GetDecisionAsync(string input, CancellationToken cancellationToken = default)
    {
        try
        {
            // Try Python model first if enabled
            if (_pythonOptions.Enabled)
            {
                var pythonResult = await CallPythonModelAsync(input, cancellationToken).ConfigureAwait(false);
                if (!string.IsNullOrWhiteSpace(pythonResult))
                {
                    _logger?.LogInformation("[DECISION_SERVICE] Python model decision: {Decision}", pythonResult);
                    return pythonResult;
                }
            }
            
            // Fallback to built-in decision logic
            await Task.Delay(100, cancellationToken).ConfigureAwait(false); // Simulate processing time
            
            // Parse input and make conservative decision
            if (string.IsNullOrWhiteSpace(input))
                return "HOLD";
            
            // Simple decision logic - always be conservative in production
            if (input.Contains("BUY") || input.Contains("SELL"))
            {
                // In production mode, this would integrate with ML models
                // For now, return HOLD for safety
                return "HOLD";
            }
            
            return "HOLD"; // Default conservative decision
        }
        catch (Exception ex)
        {
            // Log error and return safe default
            _logger?.LogError(ex, "[DECISION_SERVICE] Error in decision service: {Message}", ex.Message);
            return "HOLD";
        }
    }

    private async Task<string?> CallPythonModelAsync(string input, CancellationToken cancellationToken)
    {
        try
        {
            if (!_pythonOptions.ScriptPaths.TryGetValue("decisionService", out var scriptPath))
            {
                _logger?.LogWarning("[PYTHON] 'decisionService' script path not configured in ScriptPaths.");
                return null;
            }

            // Ensure the working directory is an absolute path
            var contentRoot = Environment.GetEnvironmentVariable("ASPNETCORE_CONTENTROOT") 
                             ?? Environment.GetEnvironmentVariable("DOTNET_CONTENTROOT") 
                             ?? AppContext.BaseDirectory;
            
            var workingDirectory = Path.GetFullPath(Path.Combine(contentRoot, _pythonOptions.WorkingDirectory));
            if (!Directory.Exists(workingDirectory))
            {
                _logger?.LogError("[PYTHON] Python working directory not found at: {WorkingDirectory}", workingDirectory);
                return null;
            }
            
            var fullScriptPath = Path.GetFullPath(Path.Combine(contentRoot, scriptPath));
            if (!File.Exists(fullScriptPath))
            {
                 _logger?.LogError("[PYTHON] Python script not found at: {FullScriptPath}", fullScriptPath);
                return null;
            }

            var startInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = _pythonOptions.PythonPath,
                Arguments = $"\"{fullScriptPath}\" --input \"{input}\"",
                WorkingDirectory = workingDirectory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };

            _logger?.LogInformation("[PYTHON] Executing: {FileName} {Arguments} in {WorkingDirectory}", startInfo.FileName, startInfo.Arguments, startInfo.WorkingDirectory);

            using var process = System.Diagnostics.Process.Start(startInfo);
            if (process == null)
            {
                _logger?.LogError("[PYTHON] Failed to start Python process.");
                return null;
            }

            var output = await process.StandardOutput.ReadToEndAsync(cancellationToken).ConfigureAwait(false);
            var error = await process.StandardError.ReadToEndAsync(cancellationToken).ConfigureAwait(false);

            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);

            if (process.ExitCode != 0)
            {
                _logger?.LogError("[PYTHON] Python script exited with code {ExitCode}. Error: {Error}", process.ExitCode, error);
                return null;
            }
            
            var result = output.Trim();
            _logger?.LogInformation("[PYTHON] Python script output: {Output}", result);
            return result;
        }
        catch (System.ComponentModel.Win32Exception ex)
        {
            _logger?.LogError(ex, "[PYTHON] Win32Exception: Error starting Python process. Is '{PythonPath}' in your system's PATH? Message: {Message}", _pythonOptions.PythonPath, ex.Message);
            return null;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "[PYTHON] An unexpected error occurred while calling the Python model: {Message}", ex.Message);
            return null;
        }
    }
}