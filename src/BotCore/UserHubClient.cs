#nullable enable
using System;
using System.Text.Json;

namespace BotCore
{
    // Lightweight event facade for user stream without pulling external dependencies into BotCore level
    public sealed class UserHubClient
    {
        public event Action<JsonElement>? OnOrder;
        public event Action<JsonElement>? OnTrade;
        public event Action<JsonElement>? OnPosition;
        public event Action<JsonElement>? OnAccount;

        // Feed methods can be called by a higher-level client to forward events here
        public void FeedOrder(JsonElement je) => SafeInvoke(OnOrder, je);
        public void FeedTrade(JsonElement je) => SafeInvoke(OnTrade, je);
        public void FeedPosition(JsonElement je) => SafeInvoke(OnPosition, je);
        public void FeedAccount(JsonElement je) => SafeInvoke(OnAccount, je);

        private static void SafeInvoke(Action<JsonElement>? evt, JsonElement arg)
        {
            try { evt?.Invoke(arg); } catch { }
        }
    }
}
