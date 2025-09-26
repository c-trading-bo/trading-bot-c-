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
                _logger.LogError(ex, "Argument error in {EventName} handler", eventName);
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "Invalid operation in {EventName} handler", eventName);
            }
            catch (NotSupportedException ex)
            {
                _logger.LogError(ex, "Not supported in {EventName} handler", eventName);
            }
            catch (ObjectDisposedException ex)
            {
                _logger.LogWarning(ex, "Object disposed during {EventName} handler execution", eventName);
            }
            catch (TimeoutException ex)
            {
                _logger.LogWarning(ex, "Timeout in {EventName} handler", eventName);
            }
            catch (OutOfMemoryException)
            {
                // Critical system exception - rethrow
                throw;
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
