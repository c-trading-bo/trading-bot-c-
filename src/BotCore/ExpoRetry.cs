using System;
using Polly;

namespace BotCore
{
    public sealed class ExpoRetry : IRetryPolicy
    {
        public TimeSpan? NextRetryDelay(RetryContext context) => context.PreviousRetryCount switch
        {
            0 => TimeSpan.FromSeconds(1),
            1 => TimeSpan.FromSeconds(2),
            2 => TimeSpan.FromSeconds(5),
            3 => TimeSpan.FromSeconds(10),
            _ => TimeSpan.FromSeconds(30),
        };
    }
}
