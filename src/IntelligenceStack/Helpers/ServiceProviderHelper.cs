using System;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

namespace TradingBot.IntelligenceStack.Helpers
{
    /// <summary>
    /// Centralized service provider helper to eliminate duplication
    /// Provides typed methods for common service resolution patterns
    /// </summary>
    internal static class ServiceProviderHelper
    {
        /// <summary>
        /// Get configuration section with proper options pattern
        /// </summary>
        public static T GetConfiguration<T>(IServiceProvider provider) where T : class
        {
            return provider.GetRequiredService<IOptions<T>>().Value;
        }

        /// <summary>
        /// Get IntelligenceStackConfig with proper validation
        /// </summary>
        public static IntelligenceStackConfig GetIntelligenceStackConfig(IServiceProvider provider)
        {
            return provider.GetRequiredService<IntelligenceStackConfig>();
        }

        /// <summary>
        /// Get ML Regime Hysteresis configuration
        /// </summary>
        public static HysteresisConfig GetMlRegimeHysteresisConfig(IServiceProvider provider)
        {
            return GetIntelligenceStackConfig(provider).ML.Regime.Hysteresis;
        }

        /// <summary>
        /// Get Promotions configuration
        /// </summary>
        public static PromotionsConfig GetPromotionsConfig(IServiceProvider provider)
        {
            return GetIntelligenceStackConfig(provider).Promotions;
        }

        /// <summary>
        /// Get ML Quarantine configuration
        /// </summary>
        public static QuarantineConfig GetQuarantineConfig(IServiceProvider provider)
        {
            return GetIntelligenceStackConfig(provider).ML.Quarantine;
        }

        /// <summary>
        /// Get SLO configuration
        /// </summary>
        public static SloConfig GetSloConfig(IServiceProvider provider)
        {
            return GetIntelligenceStackConfig(provider).SLO;
        }

        /// <summary>
        /// Get Observability configuration
        /// </summary>
        public static ObservabilityConfig GetObservabilityConfig(IServiceProvider provider)
        {
            return GetIntelligenceStackConfig(provider).Observability;
        }

        /// <summary>
        /// Get RL configuration
        /// </summary>
        public static RLConfig GetRlConfig(IServiceProvider provider)
        {
            return GetIntelligenceStackConfig(provider).RL;
        }

        /// <summary>
        /// Get Idempotent Orders configuration
        /// </summary>
        public static IdempotentConfig GetIdempotentConfig(IServiceProvider provider)
        {
            return GetIntelligenceStackConfig(provider).Orders.Idempotent;
        }

        /// <summary>
        /// Get Network configuration
        /// </summary>
        public static NetworkConfig GetNetworkConfig(IServiceProvider provider)
        {
            return GetIntelligenceStackConfig(provider).Network;
        }

        /// <summary>
        /// Get Historical configuration
        /// </summary>
        public static HistoricalConfig GetHistoricalConfig(IServiceProvider provider)
        {
            return GetIntelligenceStackConfig(provider).Historical;
        }
    }
}