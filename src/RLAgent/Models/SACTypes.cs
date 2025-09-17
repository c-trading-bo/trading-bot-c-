using System;
using System.Collections.Generic;
using TradingBot.Abstractions;

namespace TradingBot.RLAgent.Algorithms;

/// <summary>
/// Configuration for Soft Actor-Critic algorithm
/// </summary>
public class SACConfig
{
    public int StateDimension { get; set; } = 10;
    public int ActionDimension { get; set; } = 3;
    public int HiddenDimension { get; set; } = 256;
    public double LearningRateActor { get; set; } = 3e-4;
    public double LearningRateCritic { get; set; } = 3e-4;
    public double LearningRateValue { get; set; } = 3e-4;
    public double Gamma { get; set; } = 0.99;
    public double Tau { get; set; } = 0.005;
    public double Alpha { get; set; } = 0.2;
    public int BatchSize { get; set; } = 256;
    public int BufferSize { get; set; } = 1000000;
    public int UpdateFrequency { get; set; } = 1;
    public int TargetUpdateFrequency { get; set; } = 1;
    public bool AutoTuneAlpha { get; set; } = true;
    
    // Additional properties needed by the implementation
    public double TemperatureAlpha { get; set; } = 0.2;
    public int MinBufferSize { get; set; } = 1000;
    public double ActionLowBound { get; set; } = -1.0;
    public double ActionHighBound { get; set; } = 1.0;
}

/// <summary>
/// State representation for SAC algorithm
/// </summary>
public class SACState
{
    public double[] Features { get; set; } = Array.Empty<double>();
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public double Price { get; set; }
    public double Volume { get; set; }
    public Dictionary<string, double> TechnicalIndicators { get; set; } = new();
    
    public int Dimension => Features.Length;
    
    public SACState()
    {
        Timestamp = DateTime.UtcNow;
    }
    
    public SACState(double[] features) : this()
    {
        Features = (double[])features.Clone();
    }
    
    /// <summary>
    /// Propose a position fraction based on current state and market features
    /// </summary>
    public double ProposeFraction(double[] marketFeatures, RegimeType regime)
    {
        // Use technical indicators and market features to propose a position size fraction
        var rsi = TechnicalIndicators.GetValueOrDefault("RSI", 50.0);
        var volatility = TechnicalIndicators.GetValueOrDefault("Volatility", 0.1);
        
        // Base proposal based on regime
        var baseProposal = regime switch
        {
            RegimeType.Trend => 0.5,
            RegimeType.Range => 0.3,
            RegimeType.Volatility => 0.2,
            RegimeType.HighVol => 0.15,
            RegimeType.LowVol => 0.4,
            _ => 0.4
        };
        
        // Calculate base fraction based on RSI (mean reversion)
        var rsiFraction = (rsi > 70) ? -0.3 : (rsi < 30) ? 0.3 : 0.0;
        
        // Adjust for momentum using market features
        var momentumFraction = marketFeatures.Length > 0 ? Math.Sign(marketFeatures[0]) * Math.Min(Math.Abs(marketFeatures[0]) * 0.5, 0.2) : 0.0;
        
        // Reduce size in high volatility
        var volatilityAdjustment = Math.Max(0.1, 1.0 - volatility * 2.0);
        
        var totalFraction = (baseProposal + rsiFraction + momentumFraction) * volatilityAdjustment;
        
        // Clamp to reasonable bounds
        return Math.Max(-0.5, Math.Min(0.5, totalFraction));
    }
}

/// <summary>
/// Training result for SAC algorithm
/// </summary>
public class SACTrainingResult
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public int EpisodesCompleted { get; set; }
    public double AverageReward { get; set; }
    public double ActorLoss { get; set; }
    public double CriticLoss { get; set; }
    public double CriticLoss1 { get; set; }
    public double CriticLoss2 { get; set; }
    public double ValueLoss { get; set; }
    public double EntropyLoss { get; set; }
    public double Alpha { get; set; }
    public double Entropy { get; set; }
    public int BufferSize { get; set; }
    public int TotalSteps { get; set; }
    public TimeSpan TrainingDuration { get; set; }
    public Dictionary<string, object> Metrics { get; set; } = new();
    
    public SACTrainingResult()
    {
        Success = false;
        Message = "Training not completed";
    }
    
    public SACTrainingResult(bool success, string message) : this()
    {
        Success = success;
        Message = message;
    }
}

/// <summary>
/// Statistics for SAC algorithm performance
/// </summary>
public class SACStatistics
{
    public int TotalEpisodes { get; set; }
    public int TotalSteps { get; set; }
    public double AverageReward { get; set; }
    public double BestReward { get; set; }
    public double WorstReward { get; set; }
    public double AverageEpisodeLength { get; set; }
    public double CurrentLoss { get; set; }
    public double CurrentEntropy { get; set; }
    public double CurrentAlpha { get; set; }
    public double Entropy { get; set; }
    public int BufferSize { get; set; }
    public int MaxBufferSize { get; set; }
    public TimeSpan TotalTrainingTime { get; set; }
    public DateTime LastUpdateTime { get; set; }
    public Dictionary<string, double> NetworkLosses { get; set; } = new();
    public List<double> RewardHistory { get; set; } = new();
    
    public SACStatistics()
    {
        LastUpdateTime = DateTime.UtcNow;
        NetworkLosses = new Dictionary<string, double>
        {
            ["Actor"] = 0.0,
            ["Critic1"] = 0.0,
            ["Critic2"] = 0.0,
            ["Value"] = 0.0
        };
    }
}