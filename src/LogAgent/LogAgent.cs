// Agent: LogAgent
// Role: Centralized logging and diagnostics for all bot components.
// Integration: Used by all agents for structured logging and error reporting.
using System;
using System.IO;

namespace LogAgent
{
    public static class LogAgent
    {
        public static void Log(string message)
        {
            var logLine = $"[{DateTime.UtcNow:O}] {message}";
            Console.WriteLine(logLine);
            File.AppendAllText("bot.log", logLine + Environment.NewLine);
        }
    }
}
