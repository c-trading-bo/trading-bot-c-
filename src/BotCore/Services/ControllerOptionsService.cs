using System;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of controller options configuration
    /// Replaces hardcoded decisioning parameters and UCB settings
    /// </summary>
    public class ControllerOptionsService : IControllerOptionsService
    {
        private readonly IConfiguration _config;
        private readonly ILogger<ControllerOptionsService> _logger;

        public ControllerOptionsService(IConfiguration config, ILogger<ControllerOptionsService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public (double Lower, double Upper) GetConfidenceBands(string regimeType) => regimeType?.ToLower() switch
        {
            "bull" => (
                _config.GetValue("Controller:ConfidenceBands:Bull:Lower", 0.6),
                _config.GetValue("Controller:ConfidenceBands:Bull:Upper", 0.9)
            ),
            "bear" => (
                _config.GetValue("Controller:ConfidenceBands:Bear:Lower", 0.65),
                _config.GetValue("Controller:ConfidenceBands:Bear:Upper", 0.85)
            ),
            "sideways" => (
                _config.GetValue("Controller:ConfidenceBands:Sideways:Lower", 0.7),
                _config.GetValue("Controller:ConfidenceBands:Sideways:Upper", 0.8)
            ),
            _ => (
                _config.GetValue("Controller:ConfidenceBands:Default:Lower", 0.65),
                _config.GetValue("Controller:ConfidenceBands:Default:Upper", 0.85)
            )
        };

        public (int PerHour, int PerDay) GetTradePacingLimits() => (
            _config.GetValue("Controller:TradePacing:MaxPerHour", 5),
            _config.GetValue("Controller:TradePacing:MaxPerDay", 20)
        );

        public int GetSafeHoldExtrasMinutes() => 
            _config.GetValue("Controller:SafeHoldExtrasMinutes", 15);

        public double GetUcbExplorationParameter() => 
            _config.GetValue("Controller:UCB:ExplorationParameter", 1.4);

        public double GetUcbConfidenceWidth() => 
            _config.GetValue("Controller:UCB:ConfidenceWidth", 0.1);

        public int GetUcbMinSamples() => 
            _config.GetValue("Controller:UCB:MinSamples", 10);

        public double GetStrategySelectionTemperature() => 
            _config.GetValue("Controller:StrategySelectionTemperature", 0.5);

        public bool EnableDynamicConfidenceAdjustment() => 
            _config.GetValue("Controller:EnableDynamicConfidenceAdjustment", true);

        public int GetConfidenceCalibrationLookbackHours() => 
            _config.GetValue("Controller:ConfidenceCalibrationLookbackHours", 24);
    }
}