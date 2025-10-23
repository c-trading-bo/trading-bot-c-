using Microsoft.Extensions.Logging;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Multi-timeframe extension for OnlineLearningSystem.
/// Tracks which timeframe signals contributed most to winning trades.
/// 
/// Phase 7: Online Learning Updates (Week 8)
/// - Record multi-timeframe state when trades enter
/// - Track which timeframe signals contributed most
/// - Update calibration per timeframe
/// - Save separate calibration tables for 5m vs 1m
/// 
/// Design principles:
/// - Non-invasive: Extension pattern doesn't modify existing OnlineLearningSystem
/// - Production-ready: Thread-safe tracking and persistence
/// - Analytical: Provides insights into timeframe contribution
/// </summary>
public class MultiTimeframeOnlineLearning
{
    private readonly ILogger<MultiTimeframeOnlineLearning> _logger;
    private readonly string _calibrationPath;
    
    // Trade tracking: TradeId -> multi-timeframe state
    private readonly ConcurrentDictionary<string, MultiTimeframeTradeState> _tradeStates = new();
    
    // Calibration tables per timeframe
    private readonly ConcurrentDictionary<string, TimeframeCalibration> _calibrations = new();
    
    // JSON serializer options
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true
    };
    
    public MultiTimeframeOnlineLearning(
        ILogger<MultiTimeframeOnlineLearning> logger,
        string calibrationPath = "data/calibration")
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _calibrationPath = calibrationPath ?? throw new ArgumentNullException(nameof(calibrationPath));
        
        // Ensure calibration directory exists
        Directory.CreateDirectory(_calibrationPath);
        
        // Load existing calibrations
        LoadCalibrations();
        
        _logger.LogInformation(
            "[MTF_LEARNING] Multi-timeframe online learning initialized");
    }
    
    /// <summary>
    /// Record multi-timeframe state when trade enters.
    /// </summary>
    /// <param name="tradeId">Trade identifier</param>
    /// <param name="symbol">Symbol traded</param>
    /// <param name="features">Multi-timeframe features at entry</param>
    /// <param name="direction">Trade direction (1=buy, -1=sell)</param>
    public void RecordTradeEntry(
        string tradeId,
        string symbol,
        Dictionary<string, double> features,
        int direction)
    {
        if (string.IsNullOrWhiteSpace(tradeId))
        {
            _logger.LogWarning("[MTF_LEARNING] Invalid trade ID");
            return;
        }
        
        try
        {
            // Extract 5m and 1m features
            var features5m = features.Where(kvp => kvp.Key.EndsWith("_5m"))
                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
            
            var features1m = features.Where(kvp => kvp.Key.EndsWith("_1m"))
                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
            
            var state = new MultiTimeframeTradeState
            {
                TradeId = tradeId,
                Symbol = symbol,
                Direction = direction,
                EntryTime = DateTimeOffset.UtcNow,
                Features5m = features5m,
                Features1m = features1m
            };
            
            _tradeStates[tradeId] = state;
            
            _logger.LogDebug(
                "[MTF_LEARNING] Recorded trade entry for {TradeId}: {Symbol} {Direction}, " +
                "{Count5m} 5m features, {Count1m} 1m features",
                tradeId, symbol, direction > 0 ? "BUY" : "SELL",
                features5m.Count, features1m.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "[MTF_LEARNING] Error recording trade entry for {TradeId}",
                tradeId);
        }
    }
    
    /// <summary>
    /// Analyze trade outcome and update timeframe calibrations.
    /// </summary>
    /// <param name="tradeId">Trade identifier</param>
    /// <param name="pnl">Trade profit/loss</param>
    /// <param name="exitTime">Trade exit time</param>
    public void AnalyzeTradeOutcome(
        string tradeId,
        decimal pnl,
        DateTimeOffset exitTime)
    {
        if (!_tradeStates.TryGetValue(tradeId, out var state))
        {
            _logger.LogWarning(
                "[MTF_LEARNING] No entry state found for trade {TradeId}",
                tradeId);
            return;
        }
        
        try
        {
            state.ExitTime = exitTime;
            state.PnL = pnl;
            state.WasWinner = pnl > 0;
            
            // Calculate timeframe contributions
            var contribution5m = CalculateTimeframeContribution(state.Features5m, state.Direction);
            var contribution1m = CalculateTimeframeContribution(state.Features1m, state.Direction);
            
            state.Contribution5m = contribution5m;
            state.Contribution1m = contribution1m;
            
            // Update calibration tables
            UpdateCalibration("5m", state.Symbol, contribution5m, state.WasWinner);
            UpdateCalibration("1m", state.Symbol, contribution1m, state.WasWinner);
            
            // Save calibrations periodically
            SaveCalibrations();
            
            _logger.LogInformation(
                "[MTF_LEARNING] Analyzed trade {TradeId}: PnL={PnL:F2}, " +
                "Contribution 5m={C5m:F3}, 1m={C1m:F3}, Winner={Winner}",
                tradeId, pnl, contribution5m, contribution1m, state.WasWinner);
            
            // Clean up old trade state
            _tradeStates.TryRemove(tradeId, out _);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "[MTF_LEARNING] Error analyzing trade outcome for {TradeId}",
                tradeId);
        }
    }
    
    /// <summary>
    /// Calculate timeframe contribution score based on features.
    /// Higher score means this timeframe provided stronger signal.
    /// </summary>
    private static double CalculateTimeframeContribution(
        Dictionary<string, double> features,
        int direction)
    {
        if (features.Count == 0)
        {
            return 0.0;
        }
        
        double contribution = 0.0;
        int count = 0;
        
        // RSI contribution
        if (features.TryGetValue(features.Keys.FirstOrDefault(k => k.Contains("rsi")) ?? "", out var rsi))
        {
            // RSI extremes suggest stronger signal
            var rsiExtreme = Math.Abs(rsi - 0.5) * 2.0; // 0 = neutral, 1 = extreme
            contribution += rsiExtreme;
            count++;
        }
        
        // MACD histogram contribution
        if (features.TryGetValue(features.Keys.FirstOrDefault(k => k.Contains("macd_histogram")) ?? "", out var macdHist))
        {
            // MACD histogram aligned with direction suggests stronger signal
            var macdContribution = direction * macdHist > 0 ? Math.Abs(macdHist) : 0.0;
            contribution += Math.Min(macdContribution, 1.0);
            count++;
        }
        
        // Volume imbalance contribution
        if (features.TryGetValue(features.Keys.FirstOrDefault(k => k.Contains("volume_imbalance")) ?? "", out var volImb))
        {
            // Volume imbalance aligned with direction suggests stronger signal
            var volContribution = direction * volImb > 0 ? Math.Abs(volImb) : 0.0;
            contribution += volContribution;
            count++;
        }
        
        // Trend slope contribution
        if (features.TryGetValue(features.Keys.FirstOrDefault(k => k.Contains("trend_slope")) ?? "", out var trend))
        {
            // Trend aligned with direction suggests stronger signal
            var trendContribution = direction * trend > 0 ? Math.Abs(trend) / 100.0 : 0.0; // Normalize from percentage
            contribution += Math.Min(trendContribution, 1.0);
            count++;
        }
        
        if (count == 0)
        {
            return 0.0;
        }
        
        return contribution / count; // Average contribution (0-1)
    }
    
    /// <summary>
    /// Update calibration table for a timeframe.
    /// </summary>
    private void UpdateCalibration(
        string timeframe,
        string symbol,
        double contribution,
        bool wasWinner)
    {
        var key = $"{timeframe}_{symbol}";
        
        var calibration = _calibrations.GetOrAdd(key, k => new TimeframeCalibration
        {
            Timeframe = timeframe,
            Symbol = symbol
        });
        
        calibration.TotalTrades++;
        
        if (wasWinner)
        {
            calibration.WinningTrades++;
            calibration.TotalWinningContribution += contribution;
        }
        else
        {
            calibration.LosingTrades++;
            calibration.TotalLosingContribution += contribution;
        }
        
        calibration.UpdatedAt = DateTimeOffset.UtcNow;
    }
    
    /// <summary>
    /// Get calibration table for a specific timeframe and symbol.
    /// </summary>
    public TimeframeCalibration? GetCalibration(string timeframe, string symbol)
    {
        var key = $"{timeframe}_{symbol}";
        return _calibrations.GetValueOrDefault(key);
    }
    
    /// <summary>
    /// Get all calibration tables.
    /// </summary>
    public Dictionary<string, TimeframeCalibration> GetAllCalibrations()
    {
        return new Dictionary<string, TimeframeCalibration>(_calibrations);
    }
    
    /// <summary>
    /// Save calibration tables to disk.
    /// </summary>
    private void SaveCalibrations()
    {
        try
        {
            foreach (var kvp in _calibrations)
            {
                var filePath = Path.Combine(_calibrationPath, $"{kvp.Key}.json");
                var json = JsonSerializer.Serialize(kvp.Value, JsonOptions);
                File.WriteAllText(filePath, json);
            }
            
            _logger.LogDebug(
                "[MTF_LEARNING] Saved {Count} calibration tables",
                _calibrations.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MTF_LEARNING] Error saving calibration tables");
        }
    }
    
    /// <summary>
    /// Load calibration tables from disk.
    /// </summary>
    private void LoadCalibrations()
    {
        try
        {
            if (!Directory.Exists(_calibrationPath))
            {
                return;
            }
            
            var files = Directory.GetFiles(_calibrationPath, "*.json");
            
            foreach (var file in files)
            {
                try
                {
                    var json = File.ReadAllText(file);
                    var calibration = JsonSerializer.Deserialize<TimeframeCalibration>(json, JsonOptions);
                    
                    if (calibration != null)
                    {
                        var key = Path.GetFileNameWithoutExtension(file);
                        _calibrations[key] = calibration;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(
                        ex,
                        "[MTF_LEARNING] Error loading calibration file: {File}",
                        file);
                }
            }
            
            _logger.LogInformation(
                "[MTF_LEARNING] Loaded {Count} calibration tables",
                _calibrations.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MTF_LEARNING] Error loading calibration tables");
        }
    }
}

/// <summary>
/// Multi-timeframe state for a trade.
/// </summary>
public class MultiTimeframeTradeState
{
    public string TradeId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public int Direction { get; set; }
    public DateTimeOffset EntryTime { get; set; }
    public DateTimeOffset? ExitTime { get; set; }
    public decimal PnL { get; set; }
    public bool WasWinner { get; set; }
    public Dictionary<string, double> Features5m { get; set; } = new();
    public Dictionary<string, double> Features1m { get; set; } = new();
    public double Contribution5m { get; set; }
    public double Contribution1m { get; set; }
}

/// <summary>
/// Timeframe calibration table.
/// </summary>
public class TimeframeCalibration
{
    public string Timeframe { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public double TotalWinningContribution { get; set; }
    public double TotalLosingContribution { get; set; }
    public DateTimeOffset UpdatedAt { get; set; }
    
    public double WinRate => TotalTrades > 0 ? (double)WinningTrades / TotalTrades : 0.0;
    public double AvgWinningContribution => WinningTrades > 0 ? TotalWinningContribution / WinningTrades : 0.0;
    public double AvgLosingContribution => LosingTrades > 0 ? TotalLosingContribution / LosingTrades : 0.0;
}
