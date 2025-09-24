namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration service for controller and decisioning parameters
    /// Replaces hardcoded confidence bands, trade pacing, and UCB parameters
    /// </summary>
    public interface IControllerOptionsService
    {
        /// <summary>
        /// AI confidence bands for different regime types
        /// </summary>
        (double Lower, double Upper) GetConfidenceBands(string regimeType);

        /// <summary>
        /// Trade pacing parameters (max trades per hour/day)
        /// </summary>
        (int PerHour, int PerDay) GetTradePacingLimits();

        /// <summary>
        /// Safe-hold extras duration in minutes
        /// </summary>
        int GetSafeHoldExtrasMinutes();

        /// <summary>
        /// UCB exploration parameter (typically 1.0-2.0)
        /// </summary>
        double GetUcbExplorationParameter();

        /// <summary>
        /// UCB confidence interval width
        /// </summary>
        double GetUcbConfidenceWidth();

        /// <summary>
        /// Minimum samples required for UCB decisions
        /// </summary>
        int GetUcbMinSamples();

        /// <summary>
        /// Strategy selection temperature for softmax (0.0-inf)
        /// </summary>
        double GetStrategySelectionTemperature();

        /// <summary>
        /// Whether to enable dynamic confidence adjustment
        /// </summary>
        bool EnableDynamicConfidenceAdjustment();

        /// <summary>
        /// Lookback period for confidence calibration in hours
        /// </summary>
        int GetConfidenceCalibrationLookbackHours();

        /// <summary>
        /// VIX maximum value for normalization
        /// </summary>
        double GetVixMaxValue();

        /// <summary>
        /// Volume ratio maximum value for normalization
        /// </summary>
        double GetVolumeRatioMaxValue();

        /// <summary>
        /// RSI maximum value for normalization
        /// </summary>
        double GetRsiMaxValue();

        /// <summary>
        /// Momentum scale factor for feature normalization
        /// </summary>
        double GetMomentumScaleFactor();

        /// <summary>
        /// Volatility maximum value for normalization
        /// </summary>
        double GetVolatilityMaxValue();

        /// <summary>
        /// VIX neutral level for analysis
        /// </summary>
        double GetVixNeutralLevel();

        /// <summary>
        /// VIX impact factor for confidence calculation
        /// </summary>
        double GetVixImpactFactor();

        /// <summary>
        /// Volume impact factor for confidence calculation
        /// </summary>
        double GetVolumeImpactFactor();

        /// <summary>
        /// Momentum impact factor for confidence calculation
        /// </summary>
        double GetMomentumImpactFactor();

        /// <summary>
        /// Noise amplitude for model simulation
        /// </summary>
        double GetNoiseAmplitude();
    }
}