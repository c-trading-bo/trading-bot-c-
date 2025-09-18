using System;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System.Diagnostics;

namespace TopstepX.Bot.Intelligence
{
    /// <summary>
    /// Integration with Local Bot Mechanic
    /// Provides health status and monitoring data to main dashboard
    /// </summary>
    public class LocalBotMechanicIntegration
    {
        private readonly string _basePath;
        private readonly string _databasePath;
        private Process? _mechanicProcess;
        
        public LocalBotMechanicIntegration(string basePath)
        {
            _basePath = basePath;
            _databasePath = Path.Combine(basePath, "Intelligence", "mechanic", "database");
        }
        
        /// <summary>
        /// Start Local Bot Mechanic in background
        /// </summary>
        public Task<bool> StartMechanicAsync()
        {
            try
            {
                var autoStartPath = Path.Combine(_basePath, "Intelligence", "mechanic", "local", "auto_start.py");
                
                if (!File.Exists(autoStartPath))
                {
                    Console.WriteLine("❌ Local Bot Mechanic not found");
                    return Task.FromResult(false);
                }
                
                var startInfo = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = autoStartPath,
                    WorkingDirectory = _basePath,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true
                };
                
                _mechanicProcess = Process.Start(startInfo);
                
                if (_mechanicProcess != null)
                {
                    Console.WriteLine($"🧠 Local Bot Mechanic started (PID: {_mechanicProcess.Id})");
                    return Task.FromResult(true);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Failed to start Local Bot Mechanic: {ex.Message}");
            }
            
            return Task.FromResult(false);
        }
        
        /// <summary>
        /// Get current mechanic status for dashboard
        /// </summary>
        public async Task<MechanicStatus> GetStatusAsync()
        {
            try
            {
                var knowledgeFile = Path.Combine(_databasePath, "knowledge.json");
                
                if (!File.Exists(knowledgeFile))
                {
                    return new MechanicStatus
                    {
                        Status = "offline",
                        HealthScore = 0,
                        IssuesCount = 0,
                        FilesTracked = 0,
                        LastScanTime = "Never",
                        AutoFixed = 0,
                        IsMonitoring = false
                    };
                }
                
                var json = await File.ReadAllTextAsync(knowledgeFile).ConfigureAwait(false);
                var data = JsonSerializer.Deserialize<JsonElement>(json);
                
                var lastScan = data.TryGetProperty("last_scan", out var scanElement) ? scanElement : new JsonElement();
                
                var issuesCount = 0;
                if (lastScan.TryGetProperty("issues_found", out var issuesElement) && issuesElement.ValueKind == JsonValueKind.Array)
                {
                    issuesCount = issuesElement.GetArrayLength();
                }
                
                var autoFixed = 0;
                if (lastScan.TryGetProperty("auto_fixed", out var fixedElement) && fixedElement.ValueKind == JsonValueKind.Array)
                {
                    autoFixed = fixedElement.GetArrayLength();
                }
                
                var filesCount = 0;
                if (data.TryGetProperty("files", out var filesElement) && filesElement.ValueKind == JsonValueKind.Object)
                {
                    filesCount = filesElement.EnumerateObject().Count();
                }
                
                var lastScanTime = "Never";
                if (lastScan.TryGetProperty("timestamp", out var timeElement) && timeElement.ValueKind == JsonValueKind.String)
                {
                    lastScanTime = timeElement.GetString() ?? "Never";
                }
                
                return new MechanicStatus
                {
                    Status = issuesCount == 0 ? "healthy" : "warning",
                    HealthScore = Math.Max(0, 100 - (issuesCount * 10)),
                    IssuesCount = issuesCount,
                    FilesTracked = filesCount,
                    LastScanTime = lastScanTime,
                    AutoFixed = autoFixed,
                    IsMonitoring = _mechanicProcess?.HasExited == false
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ Failed to get mechanic status: {ex.Message}");
                return new MechanicStatus
                {
                    Status = "error",
                    HealthScore = 0,
                    IssuesCount = -1,
                    FilesTracked = 0,
                    LastScanTime = "Error",
                    AutoFixed = 0,
                    IsMonitoring = false
                };
            }
        }
        
        /// <summary>
        /// Stop the mechanic process
        /// </summary>
        public void Stop()
        {
            try
            {
                if (_mechanicProcess != null && !_mechanicProcess.HasExited)
                {
                    _mechanicProcess.Kill();
                    Console.WriteLine("🛑 Local Bot Mechanic stopped");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ Error stopping mechanic: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Check if mechanic is running
        /// </summary>
        public bool IsRunning => _mechanicProcess != null && !_mechanicProcess.HasExited;
    }
    
    /// <summary>
    /// Status information from Local Bot Mechanic
    /// </summary>
    public class MechanicStatus
    {
        public string Status { get; set; } = "unknown";
        public int HealthScore { get; set; }
        public int IssuesCount { get; set; }
        public int FilesTracked { get; set; }
        public string LastScanTime { get; set; } = "Never";
        public int AutoFixed { get; set; }
        public bool IsMonitoring { get; set; }
    }
}
