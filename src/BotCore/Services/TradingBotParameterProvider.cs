using Microsoft.Extensions.DependencyInjection;
using TradingBot.BotCore.Services;
using Microsoft.Extensions.Logging;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Modern replacement for OrchestratorAgent.Configuration.MLParameterProvider
    /// Provides ML configuration parameters for production trading
    /// All values are configuration-driven with no hardcoded business logic
    /// </summary>
    public static class TradingBotParameterProvider
    {
        private static IServiceProvider? _serviceProvider;
        private static ILogger? _logger;
        
        /// <summary>
        /// Initialize the provider with the service provider
        /// </summary>
        public static void Initialize(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
            _logger = serviceProvider.GetService<ILogger<MLConfigurationService>>();
        }
        
        /// <summary>
        /// Get AI confidence threshold - replaces hardcoded values in legacy code
        /// </summary>
        public static double GetAIConfidenceThreshold()
        {
            if (_serviceProvider == null)
            {
                // Conservative fallback when service provider not available
                return 0.75; 
            }
            
            using var scope = _serviceProvider.CreateScope();
            var mlConfig = scope.ServiceProvider.GetService<MLConfigurationService>();
            var threshold = mlConfig?.GetAIConfidenceThreshold() ?? 0.75;
            
            _logger?.LogDebug("Retrieved AI confidence threshold: {Threshold}", threshold);
            return threshold;
        }
        
        /// <summary>
        /// Get position size multiplier - replaces hardcoded values in legacy code
        /// </summary>
        public static double GetPositionSizeMultiplier()
        {
            if (_serviceProvider == null)
            {
                // Conservative fallback when service provider not available
                return 2.0; 
            }
            
            using var scope = _serviceProvider.CreateScope();
            var mlConfig = scope.ServiceProvider.GetService<MLConfigurationService>();
            var multiplier = mlConfig?.GetPositionSizeMultiplier() ?? 2.0;
            
            _logger?.LogDebug("Retrieved position size multiplier: {Multiplier}", multiplier);
            return multiplier;
        }
        
        /// <summary>
        /// Get fallback confidence - replaces hardcoded values in error scenarios
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
            var confidence = mlConfig?.GetMinimumConfidence() ?? 0.65;
            
            _logger?.LogDebug("Retrieved fallback confidence: {Confidence}", confidence);
            return confidence;
        }

        /// <summary>
        /// Get regime detection threshold - replaces hardcoded values
        /// </summary>
        public static double GetRegimeDetectionThreshold()
        {
            if (_serviceProvider == null)
            {
                return 1.0;
            }
            
            using var scope = _serviceProvider.CreateScope();
            var mlConfig = scope.ServiceProvider.GetService<MLConfigurationService>();
            var threshold = mlConfig?.GetRegimeDetectionThreshold() ?? 1.0;
            
            _logger?.LogDebug("Retrieved regime detection threshold: {Threshold}", threshold);
            return threshold;
        }
    }
}