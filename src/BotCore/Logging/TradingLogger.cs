// src/BotCore/Logging/TradingLogger.cs
using System;
using Microsoft.Extensions.Logging;

namespace BotCore.Logging;

public class TradingLogger : ITradingLogger
{
    private readonly ILogger<TradingLogger> _logger;

    public TradingLogger(ILogger<TradingLogger> logger)
    {
        _logger = logger;
    }

    public void LogSystem(string message)
    {
        _logger.LogInformation($"[SYS] {message}");
    }

    public void LogSuccess(string message)
    {
        // Using Information level but with a specific prefix for success
        _logger.LogInformation($"[OK] ✅ {message}");
    }

    public void LogError(string message, Exception? ex = null)
    {
        if (ex != null)
        {
            _logger.LogError(ex, $"[FAIL] ❌ {message}");
        }
        else
        {
            _logger.LogError($"[FAIL] ❌ {message}");
        }
    }

    public void LogWarning(string message)
    {
        _logger.LogWarning($"[WARN] ⚠️ {message}");
    }

    public void LogJson(string message)
    {
        // A specific method to log raw JSON if needed, at a lower level
        _logger.LogDebug(message);
    }
}
