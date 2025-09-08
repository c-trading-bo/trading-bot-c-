using System.Diagnostics;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// 🐍 PYTHON UCB SERVICE LAUNCHER 🐍
/// 
/// Automatically starts the Python UCB FastAPI service as part of UnifiedOrchestrator startup.
/// No more manual launches - everything starts together as one unified system!
/// </summary>
public sealed class PythonUcbLauncher : BackgroundService
{
    private readonly ILogger<PythonUcbLauncher> _logger;
    private Process? _pythonProcess;
    private readonly string _ucbPath;
    private readonly string _ucbPort;
    private readonly bool _enabled;

    public PythonUcbLauncher(ILogger<PythonUcbLauncher> logger)
    {
        _logger = logger;
        
        // Configuration from environment
        _enabled = Environment.GetEnvironmentVariable("ENABLE_UCB") != "0"; // Default enabled
        _ucbPort = Environment.GetEnvironmentVariable("UCB_PORT") ?? "5000";
        
        // Find UCB directory - try multiple locations
        var workspaceRoot = FindWorkspaceRoot();
        _ucbPath = Path.Combine(workspaceRoot, "python", "ucb");
        
        _logger.LogInformation("🐍 PythonUcbLauncher initialized:");
        _logger.LogInformation("   • Enabled: {Enabled}", _enabled);
        _logger.LogInformation("   • Port: {Port}", _ucbPort);
        _logger.LogInformation("   • UCB Path: {UcbPath}", _ucbPath);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!_enabled)
        {
            _logger.LogInformation("⚠️ UCB service disabled - Set ENABLE_UCB=1 to enable");
            return;
        }

        if (!Directory.Exists(_ucbPath))
        {
            _logger.LogError("❌ UCB directory not found: {UcbPath}", _ucbPath);
            return;
        }

        try
        {
            await StartPythonUcbService(stoppingToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to start Python UCB service");
        }
    }

    private async Task StartPythonUcbService(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🚀 Starting Python UCB FastAPI service...");

        // Check if port is already in use
        if (await IsPortInUse(_ucbPort))
        {
            _logger.LogWarning("⚠️ Port {Port} already in use - UCB service may already be running", _ucbPort);
            return;
        }

        // Prepare startup script path
        var scriptPath = Path.Combine(_ucbPath, "start_ucb_api.bat");
        if (!File.Exists(scriptPath))
        {
            _logger.LogError("❌ UCB startup script not found: {ScriptPath}", scriptPath);
            return;
        }

        // Start Python UCB service process
        var startInfo = new ProcessStartInfo
        {
            FileName = "cmd.exe",
            Arguments = $"/c \"{scriptPath}\"",
            WorkingDirectory = _ucbPath,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
            Environment =
            {
                ["UCB_HOST"] = "127.0.0.1",
                ["UCB_PORT"] = _ucbPort,
                ["UCB_PERSISTENCE_PATH"] = "ucb_state.pkl",
                ["UCB_WEIGHTS_PATH"] = "neural_ucb_topstep.pth"
            }
        };

        _pythonProcess = new Process { StartInfo = startInfo };
        
        // Handle output for monitoring
        _pythonProcess.OutputDataReceived += (sender, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                _logger.LogInformation("🐍 UCB: {Output}", e.Data);
            }
        };
        
        _pythonProcess.ErrorDataReceived += (sender, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                _logger.LogWarning("🐍 UCB Error: {Error}", e.Data);
            }
        };

        _pythonProcess.Start();
        _pythonProcess.BeginOutputReadLine();
        _pythonProcess.BeginErrorReadLine();

        _logger.LogInformation("✅ Python UCB service started - PID: {ProcessId}", _pythonProcess.Id);
        _logger.LogInformation("🌐 UCB FastAPI available at: http://127.0.0.1:{Port}", _ucbPort);

        // Wait for process to finish or cancellation
        try
        {
            await _pythonProcess.WaitForExitAsync(cancellationToken);
            _logger.LogInformation("🛑 Python UCB service exited - Code: {ExitCode}", _pythonProcess.ExitCode);
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("🛑 Python UCB service stopping due to cancellation");
        }
    }

    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🛑 Stopping Python UCB service...");

        if (_pythonProcess != null && !_pythonProcess.HasExited)
        {
            try
            {
                // Try graceful shutdown first
                _pythonProcess.CloseMainWindow();
                
                // Wait a bit for graceful shutdown
                if (!_pythonProcess.WaitForExit(5000))
                {
                    _logger.LogWarning("⚠️ Forcing Python UCB service termination");
                    _pythonProcess.Kill();
                }
                
                _logger.LogInformation("✅ Python UCB service stopped");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error stopping Python UCB service");
            }
            finally
            {
                _pythonProcess?.Dispose();
            }
        }

        await base.StopAsync(cancellationToken);
    }

    private static string FindWorkspaceRoot()
    {
        var current = Directory.GetCurrentDirectory();
        
        // Look for workspace indicators
        while (current != null)
        {
            if (Directory.Exists(Path.Combine(current, "python", "ucb")) ||
                File.Exists(Path.Combine(current, "Directory.Build.props")))
            {
                return current;
            }
            
            var parent = Directory.GetParent(current);
            current = parent?.FullName;
        }
        
        // Fallback to current directory
        return Directory.GetCurrentDirectory();
    }

    private static async Task<bool> IsPortInUse(string port)
    {
        try
        {
            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromSeconds(2);
            
            var response = await client.GetAsync($"http://127.0.0.1:{port}/health");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false; // Port not in use or service not responding
        }
    }

    public override void Dispose()
    {
        _pythonProcess?.Dispose();
        base.Dispose();
    }
}
