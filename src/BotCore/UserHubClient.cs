#nullable enable
using System;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace BotCore
{
    // Lightweight event facade for user stream without pulling external dependencies into BotCore level
    public sealed class UserHubClient
    {
        private static readonly ILogger _logger = 
            Microsoft.Extensions.Logging.LoggerFactory.Create(builder => builder.AddConsole())
                .CreateLogger<UserHubClient>();

        // LoggerMessage delegates for improved performance (CA1848 compliance)
        private static readonly Action<ILogger, string, Exception> LogArgumentError =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(3001), 
                "Argument error in {EventName} handler");

        private static readonly Action<ILogger, string, Exception> LogObjectDisposed =
            LoggerMessage.Define<string>(LogLevel.Warning, new EventId(3002), 
                "Object disposed during {EventName} handler execution");

        private static readonly Action<ILogger, string, Exception> LogInvalidOperation =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(3003), 
                "Invalid operation in {EventName} handler");

        private static readonly Action<ILogger, string, Exception> LogNotSupported =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(3004), 
                "Not supported in {EventName} handler");

        private static readonly Action<ILogger, string, Exception> LogTimeout =
            LoggerMessage.Define<string>(LogLevel.Warning, new EventId(3005), 
                "Timeout in {EventName} handler");

        private static readonly Action<ILogger, Exception> LogOutOfMemory =
            LoggerMessage.Define(LogLevel.Critical, new EventId(3006), 
                "Critical: Out of memory during event handler execution");

        public event Action<JsonElement>? OnOrder;
        public event Action<JsonElement>? OnTrade;
        public event Action<JsonElement>? OnPosition;
        public event Action<JsonElement>? OnAccount;

        // Feed methods can be called by a higher-level client to forward events here
        public void FeedOrder(JsonElement je) => SafeInvoke(OnOrder, je, nameof(OnOrder));
        public void FeedTrade(JsonElement je) => SafeInvoke(OnTrade, je, nameof(OnTrade));
        public void FeedPosition(JsonElement je) => SafeInvoke(OnPosition, je, nameof(OnPosition));
        public void FeedAccount(JsonElement je) => SafeInvoke(OnAccount, je, nameof(OnAccount));

        private static void SafeInvoke(Action<JsonElement>? evt, JsonElement arg, string eventName)
        {
            if (evt == null) return;
            
            try 
            { 
                evt.Invoke(arg); 
            } 
            catch (ArgumentException ex)
            {
                LogArgumentError(_logger, eventName, ex);
            }
            catch (ObjectDisposedException ex)
            {
                LogObjectDisposed(_logger, eventName, ex);
            }
            catch (InvalidOperationException ex)
            {
                LogInvalidOperation(_logger, eventName, ex);
            }
            catch (NotSupportedException ex)
            {
                LogNotSupported(_logger, eventName, ex);
            }
            catch (TimeoutException ex)
            {
                LogTimeout(_logger, eventName, ex);
            }
            catch (OutOfMemoryException ex)
            {
                LogOutOfMemory(_logger, ex);
                throw; // Critical exception, re-throw
            }
            catch (StackOverflowException)
            {
                // Critical system exception - rethrow
                throw;
            }
            catch (Exception ex) when (!(ex is SystemException))
            {
                _logger.LogError(ex, "Unexpected application error in {EventName} handler - continuing", eventName);
            }
        }
    }
}
