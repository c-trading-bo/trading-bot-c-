using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;

namespace OrchestratorAgent.Intelligence
{
    /// <summary>
    /// Local Bot Mechanic Integration
    /// Automatically starts and manages the Python-based Local Bot Mechanic
    /// </summary>
    public class LocalBotMechanicIntegration : IHostedService, IDisposable
    {
        private readonly ILogger<LocalBotMechanicIntegration> _logger;
        private Process? _mechanicProcess;
        private Process? _dashboardProcess;
        private readonly string _mechanicPath = null!;
        private bool _disposed;

        public LocalBotMechanicIntegration(ILogger<LocalBotMechanicIntegration> logger)
        {
            _logger = logger;
            _mechanicPath = Path.Combine(Directory.GetCurrentDirectory(), "Intelligence", "mechanic", "local");
        }

        public async Task StartAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("🚀 Starting Local Bot Mechanic Integration...");

            try
            {
                // Check if mechanic files exist
                if (!CheckMechanicFiles())
                {
                    _logger.LogWarning("⚠️ Local Bot Mechanic files not found - running without self-healing");
                    return;
                }

                // Start the auto-launcher
                await StartMechanicAsync().ConfigureAwait(false);

                _logger.LogInformation("✅ Local Bot Mechanic started successfully");
                _logger.LogInformation("📊 Dashboard: http://localhost:5051/mechanic/dashboard");
                _logger.LogInformation("🔗 Health API: http://localhost:5051/mechanic/health");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to start Local Bot Mechanic");
            }
        }

        public async Task StopAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("🛑 Stopping Local Bot Mechanic...");

            try
            {
                // Stop processes gracefully
                await StopProcessAsync(_mechanicProcess, "Bot Mechanic").ConfigureAwait(false);
                await StopProcessAsync(_dashboardProcess, "Dashboard").ConfigureAwait(false);

                _logger.LogInformation("✅ Local Bot Mechanic stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error stopping Local Bot Mechanic");
            }
        }

        private bool CheckMechanicFiles()
        {
            var requiredFiles = new[]
            {
                Path.Combine(_mechanicPath, "bot_mechanic.py"),
                Path.Combine(_mechanicPath, "mechanic_dashboard.py"),
                Path.Combine(_mechanicPath, "start_local_mechanic.py")
            };

            foreach (var file in requiredFiles)
            {
                if (!File.Exists(file))
                {
                    _logger.LogWarning($"Missing required file: {file}");
                    return false;
                }
            }

            return true;
        }

        private async Task StartMechanicAsync()
        {
            var startupScript = Path.Combine(_mechanicPath, "start_local_mechanic.py");
            
            if (!File.Exists(startupScript))
            {
                throw new FileNotFoundException($"Startup script not found: {startupScript}");
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = "python",
                Arguments = $"\"{startupScript}\"",
                WorkingDirectory = Directory.GetCurrentDirectory(),
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            // Set environment variables
            startInfo.Environment["PYTHONPATH"] = _mechanicPath;

            _mechanicProcess = new Process { StartInfo = startInfo };

            // Handle output
            _mechanicProcess.OutputDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    _logger.LogInformation($"[Mechanic] {e.Data}");
                }
            };

            _mechanicProcess.ErrorDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data))
                {
                    _logger.LogWarning($"[Mechanic Error] {e.Data}");
                }
            };

            try
            {
                _mechanicProcess.Start();
                _mechanicProcess.BeginOutputReadLine();
                _mechanicProcess.BeginErrorReadLine();

                // Give it time to start
                await Task.Delay(3000).ConfigureAwait(false);

                if (_mechanicProcess.HasExited)
                {
                    throw new InvalidOperationException($"Mechanic process exited immediately with code {_mechanicProcess.ExitCode}");
                }

                _logger.LogInformation("✅ Local Bot Mechanic process started");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to start mechanic process");
                throw;
            }
        }

        private async Task StopProcessAsync(Process? process, string processName)
        {
            if (process == null || process.HasExited)
                return;

            try
            {
                _logger.LogInformation($"Stopping {processName}...");

                // Try graceful shutdown first
                process.CloseMainWindow();
                
                if (!process.WaitForExit(5000))
                {
                    // Force kill if needed
                    process.Kill();
                    await process.WaitForExitAsync().ConfigureAwait(false);
                }

                _logger.LogInformation($"✅ {processName} stopped");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, $"Error stopping {processName}");
            }
        }

        /// <summary>
        /// Get health status from the mechanic
        /// </summary>
        public async Task<string?> GetHealthStatusAsync()
        {
            try
            {
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromSeconds(5);
                
                var response = await client.GetStringAsync("http://localhost:5051/mechanic/health").ConfigureAwait(false);
                return response;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to get mechanic health status");
                return null;
            }
        }

        /// <summary>
        /// Trigger a full scan through the mechanic API
        /// </summary>
        public async Task<bool> TriggerScanAsync()
        {
            try
            {
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromSeconds(30);
                
                var response = await client.GetAsync("http://localhost:5051/mechanic/scan").ConfigureAwait(false);
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to trigger mechanic scan");
                return false;
            }
        }

        /// <summary>
        /// Trigger auto-fix through the mechanic API
        /// </summary>
        public async Task<bool> TriggerAutoFixAsync()
        {
            try
            {
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromSeconds(30);
                
                var response = await client.GetAsync("http://localhost:5051/mechanic/fix").ConfigureAwait(false);
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to trigger auto-fix");
                return false;
            }
        }

        /// <summary>
        /// Get ultimate metrics from the cloud mechanic
        /// </summary>
        public async Task<string?> GetUltimateMetricsAsync()
        {
            try
            {
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromSeconds(30);
                
                var response = await client.GetAsync("http://localhost:5051/mechanic/ultimate-metrics").ConfigureAwait(false);
                if (response.IsSuccessStatusCode)
                {
                    return await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                }
                return null;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to get ultimate metrics");
                return null;
            }
        }

        /// <summary>
        /// Trigger intelligent workflow preparation
        /// </summary>
        public async Task<bool> TriggerIntelligentPreparationAsync()
        {
            try
            {
                using var client = new HttpClient();
                client.Timeout = TimeSpan.FromMinutes(5); // Longer timeout for preparation
                
                var response = await client.PostAsync("http://localhost:5051/mechanic/prepare-workflows", null).ConfigureAwait(false);
                return response.IsSuccessStatusCode;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to trigger intelligent preparation");
                return false;
            }
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            try
            {
                _mechanicProcess?.Kill();
                _mechanicProcess?.Dispose();
                
                _dashboardProcess?.Kill();
                _dashboardProcess?.Dispose();
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error disposing Local Bot Mechanic Integration");
            }

            _disposed = true;
        }
    }
}
