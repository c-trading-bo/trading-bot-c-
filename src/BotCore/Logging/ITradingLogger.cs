// src/BotCore/Logging/ITradingLogger.cs
using System;

namespace BotCore.Logging;

public interface ITradingLogger
{
    void LogSystem(string message);
    void LogSuccess(string message);
    void LogError(string message, Exception? ex = null);
    void LogWarning(string message);
    void LogJson(string message);
}
