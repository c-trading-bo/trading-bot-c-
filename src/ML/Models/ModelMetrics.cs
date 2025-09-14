using System;

namespace TradingBot.ML.Models;

/// <summary>
/// Model performance metrics
/// </summary>
public class ModelMetrics
{
    public string ModelName { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public DateTime TrainingDate { get; set; }
    public double Accuracy { get; set; }
    public double Precision { get; set; }
    public double Recall { get; set; }
    public double F1Score { get; set; }
    public double SharpeRatio { get; set; }
    public double MaxDrawdown { get; set; }
    public double WinRate { get; set; }
    public int TotalTrades { get; set; }
    public double ProfitFactor { get; set; }
    public string ValidationStatus { get; set; } = "Unknown";
    
    // Additional properties needed by HistoricalTrainer
    public double AUC { get; set; }
    public int SampleSize { get; set; }
    public double SortinoRatio { get; set; }
    public double CalmarRatio { get; set; }
    public double Alpha { get; set; }
    public double Beta { get; set; }
    public double VolatilityAnnualized { get; set; }
    public double ReturnAnnualized { get; set; }
    public double PrAt10 { get; set; } // Precision at 10
    public double RecallAt10 { get; set; } // Recall at 10
    public double F1At10 { get; set; } // F1 at 10
}