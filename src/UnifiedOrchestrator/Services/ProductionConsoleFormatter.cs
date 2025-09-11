using System;
using System.IO;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Production-ready console formatter with color coding and noise reduction
/// Shows only ERROR, WARN, and critical INFO messages with clean formatting
/// </summary>
public class ProductionConsoleFormatter : ConsoleFormatter
{
    private const string TimestampFormat = "HH:mm:ss.fff";
    
    public ProductionConsoleFormatter(IOptionsMonitor<ConsoleFormatterOptions> options)
        : base("Production")
    {
    }

    public override void Write<TState>(in LogEntry<TState> logEntry, IExternalScopeProvider? scopeProvider, TextWriter textWriter)
    {
        // Filter out framework noise
        if (ShouldSuppressLog(logEntry.Category, logEntry.LogLevel))
            return;

        var message = logEntry.Formatter(logEntry.State, logEntry.Exception);
        if (string.IsNullOrEmpty(message) && logEntry.Exception == null)
            return;

        var timestamp = DateTime.Now.ToString(TimestampFormat);
        var categoryIcon = GetCategoryIcon(logEntry.Category);
        var colorCode = GetColorCode(logEntry.LogLevel);
        var resetCode = "\u001b[0m";

        // Format: [HH:mm:ss.fff] 🔴 ERROR TradingBot.Core: Message here
        var logLine = new StringBuilder();
        logLine.Append($"[{timestamp}] ");
        logLine.Append(colorCode);
        logLine.Append(GetLevelIcon(logEntry.LogLevel));
        logLine.Append(' ');
        logLine.Append(logEntry.LogLevel.ToString().ToUpper().PadRight(5));
        logLine.Append(' ');
        logLine.Append(GetShortCategory(logEntry.Category));
        logLine.Append(": ");
        logLine.Append(message);
        logLine.Append(resetCode);

        // Add exception details for errors
        if (logEntry.Exception != null)
        {
            logLine.AppendLine();
            logLine.Append(colorCode);
            logLine.Append("      ↳ ");
            logLine.Append(logEntry.Exception.GetType().Name);
            logLine.Append(": ");
            logLine.Append(logEntry.Exception.Message);
            logLine.Append(resetCode);
        }

        textWriter.WriteLine(logLine.ToString());
    }

    private static bool ShouldSuppressLog(string category, LogLevel level)
    {
        // Suppress framework noise from Microsoft and System namespaces
        if (category.StartsWith("Microsoft.") || category.StartsWith("System."))
        {
            return level < LogLevel.Warning;
        }

        // Only show ERROR, WARN, and critical INFO messages
        return level > LogLevel.Information && level != LogLevel.Warning && level != LogLevel.Error;
    }

    private static string GetColorCode(LogLevel level) => level switch
    {
        LogLevel.Error => "\u001b[31m",      // RED
        LogLevel.Warning => "\u001b[33m",    // YELLOW  
        LogLevel.Information => "\u001b[37m", // WHITE
        LogLevel.Debug => "\u001b[37m",      // WHITE
        _ => "\u001b[32m"                    // GREEN for success
    };

    private static string GetLevelIcon(LogLevel level) => level switch
    {
        LogLevel.Error => "🔴",
        LogLevel.Warning => "🟡", 
        LogLevel.Information => "⚪",
        LogLevel.Debug => "⚪",
        _ => "🟢"
    };

    private static string GetCategoryIcon(string category)
    {
        if (category.Contains("Trading")) return "📈";
        if (category.Contains("Auth")) return "🔐";
        if (category.Contains("Hub") || category.Contains("SignalR")) return "🌐";
        if (category.Contains("ML") || category.Contains("Intelligence")) return "🤖";
        if (category.Contains("Risk")) return "⚖️";
        if (category.Contains("Market")) return "📊";
        if (category.Contains("Order")) return "📋";
        return "⚙️";
    }

    private static string GetShortCategory(string category)
    {
        // Shorten category names for console readability
        var parts = category.Split('.');
        if (parts.Length > 2)
        {
            return $"{parts[^2]}.{parts[^1]}";
        }
        return category.Length > 30 ? category[..27] + "..." : category;
    }
}