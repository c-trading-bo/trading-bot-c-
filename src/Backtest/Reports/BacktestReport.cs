using System;
using System.Collections.Generic;

namespace TradingBot.Backtest.Reports
{
    /// <summary>
    /// Backtest summary with key performance metrics
    /// Aggregated from real trade simulations
    /// </summary>
    public class BacktestSummary
    {
        public string Symbol { get; set; } = "";
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public TimeSpan Duration => EndDate - StartDate;
        
        // Capital and returns
        public decimal InitialCapital { get; set; }
        public decimal FinalCapital { get; set; }
        public decimal TotalReturn { get; set; }
        public decimal AnnualizedReturn { get; set; }
        
        // Trade statistics
        public int TotalTrades { get; set; }
        public int WinningTrades { get; set; }
        public int LosingTrades { get; set; }
        public decimal WinRate => TotalTrades > 0 ? (decimal)WinningTrades / TotalTrades : 0m;
        
        // PnL metrics
        public decimal GrossPnL { get; set; }
        public decimal TotalCommissions { get; set; }
        public decimal NetPnL { get; set; }
        public decimal AverageTrade => TotalTrades > 0 ? NetPnL / TotalTrades : 0m;
        public decimal AverageWinner { get; set; }
        public decimal AverageLoser { get; set; }
        public decimal ProfitFactor { get; set; }
        
        // Risk metrics
        public decimal MaxDrawdown { get; set; }
        public decimal SharpeRatio { get; set; }
        public decimal SortinoRatio { get; set; }
        public decimal MaxAdverseExcursion { get; set; }
        public decimal MaxFavorableExcursion { get; set; }
        
        // Execution metrics
        public decimal AverageSlippage { get; set; }
        public TimeSpan AverageTradeTime { get; set; }
        
        public Dictionary<string, object> AdditionalMetrics { get; } = new();
    }

    /// <summary>
    /// Comprehensive backtest report with detailed analysis
    /// Provides structured output for CI/CD validation gates
    /// </summary>
    public class BacktestReport
    {
        public string BacktestId { get; set; } = Guid.NewGuid().ToString();
        public string Symbol { get; set; } = "";
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public string ModelFamily { get; set; } = "";
        public string ModelId { get; set; } = "";
        public string ModelVersion { get; set; } = "";
        
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public TimeSpan ExecutionTime => EndTime - StartTime;
        
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
        
        // Capital tracking
        public decimal InitialCapital { get; set; }
        public decimal FinalCapital { get; set; }
        public decimal TotalPnL { get; set; }
        public decimal RealizedPnL { get; set; }
        public decimal UnrealizedPnL { get; set; }
        public decimal TotalCommissions { get; set; }
        public decimal TotalReturn { get; set; }
        
        // Trade statistics
        public int TotalTrades { get; set; }
        
        // Summary metrics
        public BacktestSummary Summary { get; set; } = new();
        
        // Detailed data paths for analysis
        public string MetricsPath { get; set; } = "";
        public string DecisionLogPath { get; set; } = "";
        public string FillLogPath { get; set; } = "";
        
        // Configuration used
        public Dictionary<string, object> Configuration { get; } = new();
        
        // Warnings and notes
        public List<string> Warnings { get; } = new();
        public List<string> Notes { get; } = new();
    }

    /// <summary>
    /// Walk-forward validation fold report
    /// Contains results for a single training/validation period
    /// </summary>
    public class FoldReport
    {
        public int FoldNumber { get; set; }
        public DateTime TrainingStart { get; set; }
        public DateTime TrainingEnd { get; set; }
        public DateTime ValidationStart { get; set; }
        public DateTime ValidationEnd { get; set; }
        
        public string ModelId { get; set; } = "";
        public string ModelVersion { get; set; } = "";
        
        public BacktestSummary ValidationResults { get; set; } = new();
        
        // Model metrics on training data
        public decimal TrainingAccuracy { get; set; }
        public decimal TrainingF1Score { get; set; }
        public decimal TrainingAUC { get; set; }
        
        // Model metrics on validation data  
        public decimal ValidationAccuracy { get; set; }
        public decimal ValidationF1Score { get; set; }
        public decimal ValidationAUC { get; set; }
        
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
        
        public TimeSpan TrainingTime { get; set; }
        public TimeSpan ValidationTime { get; set; }
    }

    /// <summary>
    /// Complete walk-forward validation report
    /// Aggregates results across all validation folds
    /// </summary>
    public class WfvReport
    {
        public string ReportId { get; set; } = Guid.NewGuid().ToString();
        public string Symbol { get; set; } = "";
        public string ModelFamily { get; set; } = "";
        
        public DateTime OverallStart { get; set; }
        public DateTime OverallEnd { get; set; }
        
        public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;
        public TimeSpan TotalExecutionTime { get; set; }
        
        // Validation configuration
        public int TrainingWindowDays { get; set; }
        public int ValidationWindowDays { get; set; }
        public int StepSizeDays { get; set; }
        public int PurgeDays { get; set; }
        public int EmbargoDays { get; set; }
        
        // Individual fold results
        public List<FoldReport> Folds { get; } = new();
        
        // Aggregated metrics across all folds
        public BacktestSummary AggregatedResults { get; set; } = new();
        
        // Model performance tracking
        public decimal AverageValidationAccuracy { get; set; }
        public decimal AverageValidationF1Score { get; set; }
        public decimal AverageValidationAUC { get; set; }
        
        // Stability metrics
        public decimal ReturnStandardDeviation { get; set; }
        public decimal SharpeStandardDeviation { get; set; }
        public int ConsistentlyProfitableFolds { get; set; }
        
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
        
        // CI/CD integration
        public bool PassesValidationGates { get; set; }
        public List<string> ValidationFailures { get; } = new();
        public Dictionary<string, object> CiCdMetrics { get; } = new();
    }
}