using System;
using Polly;

namespace BotCore
{
    /// <summary>
    /// Exponential backoff retry policy
    /// </summary>
    public static class ExpoRetry
    {
        /// <summary>
        /// Creates an exponential backoff retry policy
        /// </summary>
        /// <returns>Configured retry policy</returns>
        public static ResiliencePipeline CreatePolicy()
        {
            return new ResiliencePipelineBuilder()
                .AddRetry(new Polly.Retry.RetryStrategyOptions
                {
                    MaxRetryAttempts = 4,
                    BackoffType = DelayBackoffType.Exponential,
                    Delay = TimeSpan.FromSeconds(1),
                    MaxDelay = TimeSpan.FromSeconds(30)
                })
                .Build();
        }

        /// <summary>
        /// Gets delay for a specific retry attempt
        /// </summary>
        /// <param name="retryCount">The retry attempt number</param>
        /// <returns>Delay for the retry attempt</returns>
        public static TimeSpan GetRetryDelay(int retryCount) => retryCount switch
        {
            0 => TimeSpan.FromSeconds(1),
            1 => TimeSpan.FromSeconds(2),
            2 => TimeSpan.FromSeconds(5),
            3 => TimeSpan.FromSeconds(10),
            _ => TimeSpan.FromSeconds(30),
        };
    }
}
