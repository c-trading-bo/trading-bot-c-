using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace BotCore
{
    /// <summary>
    /// Enhanced training data service that collects EVERY trade for RL improvement.
    /// Integrates with live trading to capture comprehensive features and outcomes.
    /// </summary>
    public interface IEnhancedTrainingDataService
    {
        Task<string> RecordTradeAsync(TradeSignalData signalData);
        Task RecordTradeResultAsync(string tradeId, TradeOutcomeData outcomeData);
        Task<int> GetTrainingSampleCountAsync();
        Task<string?> ExportTrainingDataAsync(int minSamples = 50);
        Task CleanupOldDataAsync(int retentionDays = 7);
    }

    public class EnhancedTrainingDataService : IEnhancedTrainingDataService
    {
        private readonly ILogger<EnhancedTrainingDataService> _logger;
        private readonly string _dataPath;
        private readonly string _liveDataPath;
        private readonly List<TradeData> _currentSession = new();
        private int _tradeCounter = 0;

        public EnhancedTrainingDataService(ILogger<EnhancedTrainingDataService> logger)
        {
            _logger = logger;
            _dataPath = Path.Combine(AppContext.BaseDirectory, "data", "rl_training");
            _liveDataPath = Path.Combine(_dataPath, "live");
            
            Directory.CreateDirectory(_dataPath);
            Directory.CreateDirectory(_liveDataPath);
            
            _logger.LogInformation("[EnhancedTrainingData] Initialized - saving to {DataPath}", _liveDataPath);
        }

        public async Task<string> RecordTradeAsync(TradeSignalData signalData)
        {
            _tradeCounter++;
            var tradeId = signalData.Id ?? $"trade_{_tradeCounter}_{DateTime.UtcNow:yyyyMMdd_HHmmss}";

            // Collect comprehensive features at trade time
            var features = await GetCurrentMarketFeaturesAsync(signalData);
            
            var tradeData = new TradeData
            {
                TradeId = tradeId,
                Timestamp = DateTime.UtcNow,
                Symbol = signalData.Symbol ?? "ES",
                Action = signalData.Direction ?? "HOLD",
                Price = signalData.Entry,
                Size = signalData.Size,
                StrategyUsed = signalData.Strategy ?? "Unknown",
                StopLoss = signalData.StopLoss,
                TakeProfit = signalData.TakeProfit,
                Features = features,
                Session = DetermineSession(DateTime.UtcNow),
                Regime = signalData.Regime ?? "Range",
                Atr = signalData.Atr,
                Rsi = signalData.Rsi,
                SlipTicks = 0.1m  // Default slip estimate
            };

            _currentSession.Add(tradeData);

            // Save immediately for real-time collection
            await SaveTradeDataAsync(tradeData);

            _logger.LogInformation("[EnhancedTrainingData] Recorded trade #{TradeCounter} - {Strategy} {Action} @ {Price}",
                _tradeCounter, tradeData.StrategyUsed, tradeData.Action, tradeData.Price);

            return tradeId;
        }

        public async Task RecordTradeResultAsync(string tradeId, TradeOutcomeData outcomeData)
        {
            var trade = _currentSession.Find(t => t.TradeId == tradeId);
            if (trade != null)
            {
                // Update trade with final results
                trade.Result = outcomeData.IsWin ? "win" : "loss";
                trade.Pnl = outcomeData.ActualPnl;
                trade.ExitPrice = outcomeData.ExitPrice;
                trade.ExitTime = outcomeData.ExitTime;
                trade.HoldingTimeMinutes = outcomeData.HoldingTimeMinutes;
                trade.RMultiple = outcomeData.ActualRMultiple;
                trade.MaxDrawdown = outcomeData.MaxDrawdown;

                // Save as complete training sample
                await SaveCompleteTradeAsync(trade);

                _logger.LogInformation("[EnhancedTrainingData] Updated trade {TradeId}: {Result} (R={RMultiple:F2})",
                    tradeId, trade.Result, trade.RMultiple);
            }
            else
            {
                _logger.LogWarning("[EnhancedTrainingData] Trade {TradeId} not found in current session", tradeId);
            }
        }

        public async Task<int> GetTrainingSampleCountAsync()
        {
            var completedFile = Path.Combine(_liveDataPath, "completed_trades.jsonl");
            if (!File.Exists(completedFile))
                return 0;

            var lines = await File.ReadAllLinesAsync(completedFile);
            return lines.Length;
        }

        public async Task<string?> ExportTrainingDataAsync(int minSamples = 50)
        {
            var completedFile = Path.Combine(_liveDataPath, "completed_trades.jsonl");
            
            if (!File.Exists(completedFile))
            {
                _logger.LogWarning("[EnhancedTrainingData] No completed trades file found");
                return null;
            }

            var lines = await File.ReadAllLinesAsync(completedFile);
            
            if (lines.Length < minSamples)
            {
                _logger.LogInformation("[EnhancedTrainingData] Need {MinSamples} samples, have {CurrentSamples} - waiting for more",
                    minSamples, lines.Length);
                return null;
            }

            // Export to CSV format for RL training
            var csvData = new List<string> { CreateCsvHeader() };
            
            foreach (var line in lines)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                
                try
                {
                    var trade = JsonSerializer.Deserialize<TradeData>(line);
                    if (trade?.RMultiple != null && trade.Features?.Count >= 20)
                    {
                        csvData.Add(CreateCsvRow(trade));
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[EnhancedTrainingData] Failed to parse trade data line");
                }
            }

            if (csvData.Count > 1) // Header + data
            {
                var exportFile = Path.Combine(_liveDataPath, $"training_export_{DateTime.UtcNow:yyyyMMdd_HHmmss}.csv");
                await File.WriteAllLinesAsync(exportFile, csvData);
                
                _logger.LogInformation("[EnhancedTrainingData] Exported {SampleCount} training samples to {ExportFile}",
                    csvData.Count - 1, exportFile);
                    
                return exportFile;
            }

            return null;
        }

        public async Task CleanupOldDataAsync(int retentionDays = 7)
        {
            var cutoffDate = DateTime.UtcNow.AddDays(-retentionDays);
            var pattern = "live_trades_*.jsonl";
            
            var files = Directory.GetFiles(_liveDataPath, pattern);
            
            foreach (var file in files)
            {
                var fileInfo = new FileInfo(file);
                if (fileInfo.LastWriteTime < cutoffDate)
                {
                    File.Delete(file);
                    _logger.LogInformation("[EnhancedTrainingData] Removed old file: {FileName}", fileInfo.Name);
                }
            }
        }

        private async Task SaveTradeDataAsync(TradeData tradeData)
        {
            var dateStr = DateTime.UtcNow.ToString("yyyyMMdd");
            var filename = Path.Combine(_liveDataPath, $"live_trades_{dateStr}.jsonl");
            
            var json = JsonSerializer.Serialize(tradeData);
            await File.AppendAllTextAsync(filename, json + Environment.NewLine);
        }

        private async Task SaveCompleteTradeAsync(TradeData tradeData)
        {
            // Save to completed trades file
            var completedFile = Path.Combine(_liveDataPath, "completed_trades.jsonl");
            var json = JsonSerializer.Serialize(tradeData);
            await File.AppendAllTextAsync(completedFile, json + Environment.NewLine);

            // Also save to strategy-specific file
            var strategy = tradeData.StrategyUsed?.ToLowerInvariant() ?? "unknown";
            var strategyFile = Path.Combine(_liveDataPath, $"completed_{strategy}_trades.jsonl");
            await File.AppendAllTextAsync(strategyFile, json + Environment.NewLine);

            _logger.LogDebug("[EnhancedTrainingData] Saved complete training sample: {Result} for {Strategy}",
                tradeData.Result, tradeData.StrategyUsed);
        }

        private async Task<List<decimal>> GetCurrentMarketFeaturesAsync(TradeSignalData signalData)
        {
            // Extract features from the signal data and current market state
            var features = new List<decimal>();

            // Basic price features (use signal data)
            features.Add(signalData.Entry);
            features.Add(signalData.Entry * 1.001m); // Simulated high
            features.Add(signalData.Entry * 0.999m); // Simulated low
            features.Add(signalData.Entry);
            features.Add(signalData.Entry); // VWAP approximation

            // Technical indicators from signal
            features.Add(signalData.Rsi);
            features.Add(signalData.Atr);
            features.Add(signalData.Ema20 ?? signalData.Entry);
            features.Add(signalData.Ema50 ?? signalData.Entry);
            features.Add(signalData.BbUpper ?? signalData.Entry * 1.02m);
            features.Add(signalData.BbLower ?? signalData.Entry * 0.98m);

            // Volume and momentum (simulated for now)
            features.Add(1000000m); // Volume
            features.Add(1.5m); // Volume ratio
            features.Add(signalData.Momentum ?? 0m);
            features.Add(signalData.TrendStrength ?? 0.5m);

            // Time-based features
            var now = DateTime.Now;
            features.Add(now.Hour);
            features.Add(now.Minute);
            features.Add((decimal)now.DayOfWeek);

            // Market regime features
            features.Add(GetRegimeScore(signalData.Regime));
            features.Add(signalData.VixLevel ?? 20m);

            // Ensure we have exactly 43 features (pad with zeros if needed)
            while (features.Count < 43)
            {
                features.Add(0m);
            }

            return features.Take(43).ToList();
        }

        private static decimal GetRegimeScore(string? regime)
        {
            return regime?.ToLowerInvariant() switch
            {
                "trend" => 1.0m,
                "range" => 0.0m,
                "highvol" => -0.5m,
                "lowvol" => 0.5m,
                _ => 0.0m
            };
        }

        private static string DetermineSession(DateTime timestamp)
        {
            var hour = timestamp.Hour;
            return hour >= 9 && hour <= 16 ? "RTH" : "ETH";
        }

        private static string CreateCsvHeader()
        {
            var headers = new List<string>
            {
                "timestamp", "symbol", "session", "regime", "R_multiple", "slip_ticks"
            };
            
            // Add feature columns
            for (int i = 1; i <= 20; i++)
            {
                headers.Add($"feature_{i}");
            }
            
            return string.Join(",", headers);
        }

        private static string CreateCsvRow(TradeData trade)
        {
            var values = new List<string>
            {
                trade.Timestamp.ToString("yyyy-MM-dd HH:mm:ss"),
                trade.Symbol ?? "ES",
                trade.Session ?? "RTH",
                trade.Regime ?? "Range",
                trade.RMultiple?.ToString("F4") ?? "0",
                trade.SlipTicks.ToString("F2")
            };

            // Add first 20 features
            var features = trade.Features ?? new List<decimal>();
            for (int i = 0; i < 20; i++)
            {
                values.Add(i < features.Count ? features[i].ToString("F6") : "0");
            }

            return string.Join(",", values);
        }
    }

    // Data models for the enhanced training service
    public class TradeSignalData
    {
        public string? Id { get; set; }
        public string? Symbol { get; set; }
        public string? Direction { get; set; }
        public decimal Entry { get; set; }
        public decimal Size { get; set; }
        public string? Strategy { get; set; }
        public decimal StopLoss { get; set; }
        public decimal TakeProfit { get; set; }
        public string? Regime { get; set; }
        public decimal Atr { get; set; }
        public decimal Rsi { get; set; }
        public decimal? Ema20 { get; set; }
        public decimal? Ema50 { get; set; }
        public decimal? BbUpper { get; set; }
        public decimal? BbLower { get; set; }
        public decimal? Momentum { get; set; }
        public decimal? TrendStrength { get; set; }
        public decimal? VixLevel { get; set; }
    }

    public class TradeOutcomeData
    {
        public bool IsWin { get; set; }
        public decimal ActualPnl { get; set; }
        public decimal ExitPrice { get; set; }
        public DateTime ExitTime { get; set; }
        public decimal HoldingTimeMinutes { get; set; }
        public decimal ActualRMultiple { get; set; }
        public decimal MaxDrawdown { get; set; }
    }

    public class TradeData
    {
        public string TradeId { get; set; } = "";
        public DateTime Timestamp { get; set; }
        public string Symbol { get; set; } = "";
        public string Action { get; set; } = "";
        public decimal Price { get; set; }
        public decimal Size { get; set; }
        public string StrategyUsed { get; set; } = "";
        public decimal StopLoss { get; set; }
        public decimal TakeProfit { get; set; }
        public List<decimal> Features { get; set; } = new();
        public string Session { get; set; } = "";
        public string Regime { get; set; } = "";
        public decimal Atr { get; set; }
        public decimal Rsi { get; set; }
        public decimal SlipTicks { get; set; }
        
        // Result fields (filled when trade closes)
        public string? Result { get; set; }
        public decimal? Pnl { get; set; }
        public decimal? ExitPrice { get; set; }
        public DateTime? ExitTime { get; set; }
        public decimal? HoldingTimeMinutes { get; set; }
        public decimal? RMultiple { get; set; }
        public decimal? MaxDrawdown { get; set; }
    }
}