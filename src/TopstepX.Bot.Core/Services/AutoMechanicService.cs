using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System.Collections.Generic;

namespace TopstepX.Bot.Core.Services
{
    /// <summary>
    /// Auto-starts and manages the Local Bot Mechanic
    /// Integrates with main dashboard - no separate dashboard needed
    /// </summary>
    public class AutoMechanicService : BackgroundService
    {
        private readonly ILogger<AutoMechanicService> _logger;
        private Process? _mechanicProcess;
        private readonly string _mechanicPath = null!;
        private readonly string _statusPath = null!;
        private Timer? _statusCheckTimer;

        public AutoMechanicService(ILogger<AutoMechanicService> logger)
        {
            _logger = logger;
            _mechanicPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Intelligence", "mechanic", "auto_start_mechanic.py");
            _statusPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Intelligence", "mechanic", "database", "dashboard_status.json");
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("🚀 Starting Auto Bot Mechanic Service...");

            try
            {
                // Start the mechanic process
                await StartMechanicProcess(stoppingToken).ConfigureAwait(false);

                // Start status monitoring for dashboard
                StartStatusMonitoring();

                _logger.LogInformation("✅ Auto Bot Mechanic Service started successfully");

                // Keep running until cancellation
                await Task.Delay(Timeout.Infinite, stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("🛑 Auto Bot Mechanic Service stopping...");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error in Auto Bot Mechanic Service");
            }
        }

        private async Task StartMechanicProcess(CancellationToken cancellationToken)
        {
            try
            {
                if (!File.Exists(_mechanicPath))
                {
                    _logger.LogWarning("⚠️ Bot Mechanic not found at: {Path}", _mechanicPath);
                    return;
                }

                var processStartInfo = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = _mechanicPath,
                    WorkingDirectory = AppDomain.CurrentDomain.BaseDirectory,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                _mechanicProcess = new Process { StartInfo = processStartInfo };

                // Handle output
                _mechanicProcess.OutputDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        _logger.LogInformation("🧠 Mechanic: {Output}", e.Data);
                };

                _mechanicProcess.ErrorDataReceived += (sender, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                        _logger.LogWarning("⚠️ Mechanic: {Error}", e.Data);
                };

                _mechanicProcess.Start();
                _mechanicProcess.BeginOutputReadLine();
                _mechanicProcess.BeginErrorReadLine();

                _logger.LogInformation("✅ Bot Mechanic process started (PID: {ProcessId})", _mechanicProcess.Id);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to start Bot Mechanic process");
            }
        }

        private void StartStatusMonitoring()
        {
            // Check status every 30 seconds for dashboard integration
            _statusCheckTimer = new Timer(CheckMechanicStatus, null, TimeSpan.Zero, TimeSpan.FromSeconds(30));
        }

        private void CheckMechanicStatus(object? state)
        {
            try
            {
                if (File.Exists(_statusPath))
                {
                    var statusJson = File.ReadAllText(_statusPath);
                    var status = JsonConvert.DeserializeObject<Dictionary<string, object>>(statusJson);
                    
                    if (status != null)
                    {
                        var isHealthy = status.TryGetValue("status", out var statusValue) && 
                                       statusValue?.ToString() == "healthy";
                        
                        if (!isHealthy)
                        {
                            var issuesCount = status.TryGetValue("issues_count", out var issues) ? issues : 0;
                            _logger.LogWarning("⚠️ Bot Mechanic found {Issues} issues", issuesCount);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug("Could not read mechanic status: {Error}", ex.Message);
            }
        }

        /// <summary>
        /// Get current mechanic status for dashboard
        /// </summary>
        public async Task<object> GetStatusAsync()
        {
            try
            {
                if (File.Exists(_statusPath))
                {
                    var statusJson = await File.ReadAllTextAsync(_statusPath).ConfigureAwait(false);
                    return JsonConvert.DeserializeObject(statusJson) ?? new { status = "unknown" };
                }
            }
            catch (Exception ex)
            {
                _logger.LogDebug("Could not read mechanic status: {Error}", ex.Message);
            }

            return new { status = "not_running", healthy = false };
        }

        /// <summary>
        /// Trigger a full scan via the mechanic
        /// </summary>
        public async Task<bool> TriggerFullScanAsync()
        {
            try
            {
                var triggerFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Intelligence", "mechanic", "database", "trigger_scan.json");
                var trigger = new { action = "full_scan", timestamp = DateTime.UtcNow };
                
                await File.WriteAllTextAsync(triggerFile, JsonConvert.SerializeObject(trigger, Formatting.Indented)).ConfigureAwait(false);
                
                _logger.LogInformation("🔍 Triggered full bot scan");
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to trigger full scan");
                return false;
            }
        }

        public override async Task StopAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("🛑 Stopping Auto Bot Mechanic Service...");

            _statusCheckTimer?.Dispose();

            if (_mechanicProcess != null && !_mechanicProcess.HasExited)
            {
                try
                {
                    _mechanicProcess.Kill();
                    await _mechanicProcess.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "⚠️ Error stopping mechanic process");
                }
                finally
                {
                    _mechanicProcess?.Dispose();
                }
            }

            await base.StopAsync(cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("✅ Auto Bot Mechanic Service stopped");
        }
    }
}
