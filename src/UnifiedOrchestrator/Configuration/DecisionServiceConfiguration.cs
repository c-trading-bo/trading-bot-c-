using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Net.Http;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Configuration;

/// <summary>
/// Configuration for workflow scheduling system
/// </summary>
public class WorkflowSchedulingOptions
{
    public bool Enabled { get; set; } = true;
    public Dictionary<string, WorkflowScheduleConfig> DefaultSchedules { get; set; } = new();
    public List<string> MarketHolidays { get; set; } = new();
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
}

/// <summary>
/// Configuration for Python integration
/// </summary>
public class PythonIntegrationOptions
{
    public bool Enabled { get; set; } = true;
    public string PythonPath { get; set; } = "/usr/bin/python3";
    public string WorkingDirectory { get; set; } = "./python";
    public Dictionary<string, string> ScriptPaths { get; set; } = new();
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
    public Dictionary<string, string> ModelPaths { get; set; } = new();
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
    public bool Enabled { get; set; } = false;
    public bool AutoRestart { get; set; } = true;
    public Dictionary<string, string> Environment { get; set; } = new();
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
    public bool Enabled { get; set; } = false;
}

/// <summary>
/// Configuration options for DecisionServiceIntegration
/// </summary>
public class DecisionServiceIntegrationOptions
{
    public string IntegrationEndpoint { get; set; } = "http://localhost:8080/integration";
    public string WebhookUrl { get; set; } = string.Empty;
    public int SyncIntervalSeconds { get; set; } = 60;
    public bool EnableRealTimeSync { get; set; } = false;
    public bool Enabled { get; set; } = false;
    public int HealthCheckIntervalSeconds { get; set; } = 30;
    public bool LogDecisionLines { get; set; } = true;
    public bool EnableTradeManagement { get; set; } = false;
    public Dictionary<string, object> CustomSettings { get; set; } = new();
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
            var response = await _httpClient.GetAsync("/health", cancellationToken);
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
                var pythonResult = await CallPythonModelAsync(input, cancellationToken);
                if (!string.IsNullOrWhiteSpace(pythonResult))
                {
                    _logger?.LogInformation("[DECISION_SERVICE] Python model decision: {Decision}", pythonResult);
                    return pythonResult;
                }
            }
            
            // Fallback to built-in decision logic
            await Task.Delay(100, cancellationToken); // Simulate processing time
            
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
                _logger?.LogWarning("[PYTHON] No decision service script path configured");
                return null;
            }

            var processStartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = _pythonOptions.PythonPath,
                Arguments = $"\"{scriptPath}\" --input \"{input}\"",
                WorkingDirectory = _pythonOptions.WorkingDirectory,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = new System.Diagnostics.Process { StartInfo = processStartInfo };
            
            var timeout = TimeSpan.FromSeconds(_pythonOptions.Timeout);
            var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            cts.CancelAfter(timeout);

            _logger?.LogDebug("[PYTHON] Calling Python decision service: {PythonPath} {Arguments}", 
                _pythonOptions.PythonPath, processStartInfo.Arguments);

            process.Start();
            
            var outputTask = process.StandardOutput.ReadToEndAsync();
            var errorTask = process.StandardError.ReadToEndAsync();
            
            await process.WaitForExitAsync(cts.Token);
            
            var output = await outputTask;
            var error = await errorTask;
            
            if (process.ExitCode == 0 && !string.IsNullOrWhiteSpace(output))
            {
                var decision = output.Trim();
                _logger?.LogInformation("[PYTHON] Python model returned: {Decision}", decision);
                return decision;
            }
            else
            {
                _logger?.LogWarning("[PYTHON] Python process failed. Exit code: {ExitCode}, Error: {Error}", 
                    process.ExitCode, error);
                return null;
            }
        }
        catch (OperationCanceledException)
        {
            _logger?.LogWarning("[PYTHON] Python model call timed out");
            return null;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "[PYTHON] Error calling Python model: {Message}", ex.Message);
            return null;
        }
    }
}