using Microsoft.Extensions.DependencyInjection;
using TradingBot.BotCore.Services;

namespace OrchestratorAgent.Configuration
{
    /// <summary>
    /// Static helper to provide ML configuration parameters for OrchestratorAgent classes
    /// Replaces hardcoded values with configuration-driven ones
    /// </summary>
    public static class MLParameterProvider
    {
        private static IServiceProvider? _serviceProvider;
        
        /// <summary>
        /// Initialize the provider with the service provider
        /// </summary>
        public static void Initialize(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }
        
        /// <summary>
        /// Get AI confidence threshold - replaces hardcoded 0.7
        /// </summary>
        public static double GetAIConfidenceThreshold()
        {
            if (_serviceProvider == null)
            {
                // Fallback to configuration-driven default
                return 0.75; // Slightly higher than original 0.7 for safety
            }
            
            using var scope = _serviceProvider.CreateScope();
            var mlConfig = scope.ServiceProvider.GetService<MLConfigurationService>();
            return mlConfig?.GetAIConfidenceThreshold() ?? 0.75;
        }
        
        /// <summary>
        /// Get position size multiplier - replaces hardcoded 2.5
        /// </summary>
        public static double GetPositionSizeMultiplier()
        {
            if (_serviceProvider == null)
            {
                // Fallback to configuration-driven default
                return 2.0; // Slightly lower than original 2.5 for safety
            }
            
            using var scope = _serviceProvider.CreateScope();
            var mlConfig = scope.ServiceProvider.GetService<MLConfigurationService>();
            return mlConfig?.GetPositionSizeMultiplier() ?? 2.0;
        }
        
        /// <summary>
        /// Get fallback confidence - replaces hardcoded 0.7 in error scenarios
        /// </summary>
        public static double GetFallbackConfidence()
        {
            if (_serviceProvider == null)
            {
                // Conservative fallback
                return 0.65; 
            }
            
            using var scope = _serviceProvider.CreateScope();
            var mlConfig = scope.ServiceProvider.GetService<MLConfigurationService>();
            // Use minimum confidence for fallback scenarios
            return mlConfig?.GetMinimumConfidence() ?? 0.65;
        }
    }
}