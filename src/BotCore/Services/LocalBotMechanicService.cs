using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Services
{
    /// <summary>
    /// Simple auto-starter for Local Bot Mechanic system
    /// Runs the Python auto-start script in the background
    /// </summary>
    public class LocalBotMechanicService
    {
        private static Process? _mechanicProcess;
        private static bool _started = false;

        /// <summary>
        /// Start the Local Bot Mechanic automatically (fire and forget)
        /// </summary>
        public static void StartAutomatic()
        {
            if (_started) return;
            _started = true;

            _ = Task.Run(() =>
            {
                try
                {
                    var autoStartPath = Path.Combine(Directory.GetCurrentDirectory(), "Intelligence", "mechanic", "auto_start.py");
                    
                    if (!File.Exists(autoStartPath))
                    {
                        Console.WriteLine("ℹ️ Local Bot Mechanic auto-start not found (optional)");
                        return;
                    }

                    Console.WriteLine("🧠 Auto-starting Local Bot Mechanic...");

                    var startInfo = new ProcessStartInfo
                    {
                        FileName = "python",
                        Arguments = $"\"{autoStartPath}\"",
                        WorkingDirectory = Directory.GetCurrentDirectory(),
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };

                    _mechanicProcess = Process.Start(startInfo);
                    
                    if (_mechanicProcess != null)
                    {
                        Console.WriteLine($"✅ Local Bot Mechanic auto-started (PID: {_mechanicProcess.Id})");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"ℹ️ Local Bot Mechanic auto-start failed (optional): {ex.Message}");
                }
            });
        }

        /// <summary>
        /// Stop the Local Bot Mechanic
        /// </summary>
        public static void Stop()
        {
            try
            {
                if (_mechanicProcess != null && !_mechanicProcess.HasExited)
                {
                    Console.WriteLine("🧠 Stopping Local Bot Mechanic...");
                    _mechanicProcess.Kill(entireProcessTree: true);
                    _mechanicProcess.WaitForExit(5000);
                    _mechanicProcess.Dispose();
                    Console.WriteLine("✅ Local Bot Mechanic stopped");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️ Error stopping Local Bot Mechanic: {ex.Message}");
            }
        }
    }
}
