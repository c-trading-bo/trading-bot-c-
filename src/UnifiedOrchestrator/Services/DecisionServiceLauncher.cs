using System;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace UnifiedOrchestrator.Services
{
    /// <summary>
    /// Service to automatically launch and manage the Python Decision Service
    /// Provides integrated startup/shutdown with the C# orchestrator
    /// </summary>
    public class DecisionServiceLauncher : BackgroundService
    {
        private readonly ILogger<DecisionServiceLauncher> _logger;
        private readonly DecisionServiceLauncherOptions _options;
        private readonly IHttpClientFactory _httpClientFactory;
        private Process? _pythonProcess;
        private bool _serviceStarted = false;

        public DecisionServiceLauncher(
            ILogger<DecisionServiceLauncher> logger,
            IOptions<DecisionServiceLauncherOptions> options,
            IHttpClientFactory httpClientFactory)
        {
            _logger = logger;
            _options = options.Value;
            _httpClientFactory = httpClientFactory;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            if (!_options.Enabled)
            {
                _logger.LogInformation("🔧 Decision Service launcher disabled");
                return;
            }

            try
            {
                _logger.LogInformation("🚀 Starting Python Decision Service launcher...");

                // Check if service is already running
                if (await IsServiceRunningAsync())
                {
                    _logger.LogInformation("✅ Decision Service already running on {Host}:{Port}", 
                        _options.Host, _options.Port);
                    _serviceStarted = true;
                    return;
                }

                // Start the Python service
                await StartPythonServiceAsync(stoppingToken);

                // Wait for service to be ready
                await WaitForServiceReadyAsync(stoppingToken);

                _logger.LogInformation("✅ Decision Service ready at {BaseUrl}", GetServiceUrl());
                _serviceStarted = true;

                // Monitor service health while running
                await MonitorServiceHealthAsync(stoppingToken);
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("🛑 Decision Service launcher stopping...");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error in Decision Service launcher");
            }
            finally
            {
                await StopPythonServiceAsync();
            }
        }

        private async Task<bool> IsServiceRunningAsync()
        {
            try
            {
                using var client = _httpClientFactory.CreateClient();
                client.Timeout = TimeSpan.FromSeconds(2);

                var response = await client.GetAsync($"{GetServiceUrl()}/health");
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }

        private async Task StartPythonServiceAsync(CancellationToken cancellationToken)
        {
            var scriptPath = GetDecisionServiceScriptPath();
            
            if (!File.Exists(scriptPath))
            {
                throw new FileNotFoundException($"Decision Service script not found: {scriptPath}");
            }

            _logger.LogInformation("🐍 Starting Python Decision Service: {ScriptPath}", scriptPath);

            var startInfo = new ProcessStartInfo
            {
                FileName = _options.PythonExecutable,
                Arguments = $"\"{scriptPath}\"",
                WorkingDirectory = Path.GetDirectoryName(scriptPath),
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            // Set environment variables
            startInfo.EnvironmentVariables["DECISION_SERVICE_HOST"] = _options.Host;
            startInfo.EnvironmentVariables["DECISION_SERVICE_PORT"] = _options.Port.ToString();
            startInfo.EnvironmentVariables["DECISION_SERVICE_CONFIG"] = _options.ConfigFile;

            _pythonProcess = new Process { StartInfo = startInfo };

            // Handle output
            _pythonProcess.OutputDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    _logger.LogInformation("[Decision Service] {Output}", e.Data);
                }
            };

            _pythonProcess.ErrorDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    _logger.LogWarning("[Decision Service Error] {Error}", e.Data);
                }
            };

            _pythonProcess.Start();
            _pythonProcess.BeginOutputReadLine();
            _pythonProcess.BeginErrorReadLine();

            _logger.LogInformation("🚀 Python Decision Service started - PID: {ProcessId}", _pythonProcess.Id);
        }

        private async Task WaitForServiceReadyAsync(CancellationToken cancellationToken)
        {
            var timeout = TimeSpan.FromSeconds(_options.StartupTimeoutSeconds);
            var checkInterval = TimeSpan.FromMilliseconds(500);
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            _logger.LogInformation("⏳ Waiting for Decision Service to be ready...");

            while (stopwatch.Elapsed < timeout && !cancellationToken.IsCancellationRequested)
            {
                if (await IsServiceRunningAsync())
                {
                    _logger.LogInformation("✅ Decision Service ready in {Elapsed:F1}s", stopwatch.Elapsed.TotalSeconds);
                    return;
                }

                await Task.Delay(checkInterval, cancellationToken);
            }

            throw new TimeoutException($"Decision Service did not start within {timeout.TotalSeconds}s");
        }

        private async Task MonitorServiceHealthAsync(CancellationToken cancellationToken)
        {
            var checkInterval = TimeSpan.FromSeconds(_options.HealthCheckIntervalSeconds);

            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(checkInterval, cancellationToken);

                    // Check if Python process is still running
                    if (_pythonProcess != null && _pythonProcess.HasExited)
                    {
                        _logger.LogError("❌ Python Decision Service process has exited unexpectedly (Exit Code: {ExitCode})", 
                            _pythonProcess.ExitCode);
                        
                        if (_options.AutoRestart)
                        {
                            _logger.LogInformation("🔄 Attempting to restart Decision Service...");
                            await StartPythonServiceAsync(cancellationToken);
                            await WaitForServiceReadyAsync(cancellationToken);
                        }
                        else
                        {
                            break;
                        }
                    }

                    // Check service health
                    if (!await IsServiceRunningAsync())
                    {
                        _logger.LogWarning("⚠️ Decision Service health check failed");
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "⚠️ Error during health monitoring");
                }
            }
        }

        private async Task StopPythonServiceAsync()
        {
            if (_pythonProcess != null && !_pythonProcess.HasExited)
            {
                _logger.LogInformation("🛑 Stopping Python Decision Service...");

                try
                {
                    _pythonProcess.Kill(entireProcessTree: true);
                    
                    // Wait for graceful shutdown
                    if (!_pythonProcess.WaitForExit(5000))
                    {
                        _logger.LogWarning("⚠️ Decision Service did not stop gracefully, force terminated");
                    }
                    else
                    {
                        _logger.LogInformation("✅ Decision Service stopped gracefully");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "⚠️ Error stopping Decision Service");
                }
                finally
                {
                    _pythonProcess.Dispose();
                    _pythonProcess = null;
                }
            }
        }

        private string GetDecisionServiceScriptPath()
        {
            var basePath = _options.ScriptPath;
            
            if (string.IsNullOrEmpty(basePath))
            {
                // Default path relative to application
                basePath = Path.Combine(
                    Directory.GetCurrentDirectory(), 
                    "..", "..", "python", "decision_service", "decision_service.py"
                );
            }

            return Path.GetFullPath(basePath);
        }

        private string GetServiceUrl()
        {
            return $"http://{_options.Host}:{_options.Port}";
        }

        public bool IsServiceReady => _serviceStarted;

        public override void Dispose()
        {
            StopPythonServiceAsync().Wait(TimeSpan.FromSeconds(10));
            base.Dispose();
        }
    }

    /// <summary>
    /// Configuration options for Decision Service launcher
    /// </summary>
    public class DecisionServiceLauncherOptions
    {
        public bool Enabled { get; set; } = true;
        public string Host { get; set; } = "127.0.0.1";
        public int Port { get; set; } = 7080;
        public string PythonExecutable { get; set; } = "python";
        public string ScriptPath { get; set; } = string.Empty;
        public string ConfigFile { get; set; } = "decision_service_config.yaml";
        public int StartupTimeoutSeconds { get; set; } = 30;
        public int HealthCheckIntervalSeconds { get; set; } = 30;
        public bool AutoRestart { get; set; } = true;
    }
}