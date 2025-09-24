#nullable enable
using System;

namespace BotCore
{
    // Production helper to illustrate wiring pattern without external deps
    public static class ExampleWireUp
    {
        // Allows passing in subscribe and handler actions to wire a simple feed
        public static void Wire<T>(Action<Action<T>> subscribe, Action<T> onEvent)
        {
            if (subscribe == null || onEvent == null) return;
            subscribe(onEvent);
        }
    }
}
