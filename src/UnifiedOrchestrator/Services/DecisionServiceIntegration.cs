using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

namespace UnifiedOrchestrator.Services
{
    /// <summary>
    /// Integration service that wires the Decision Service into the trading orchestrator
    /// Handles strategy signals, order lifecycle, and position management through the ML/RL decision brain
    /// </summary>
    public class DecisionServiceIntegration : BackgroundService
    {
        private readonly ILogger<DecisionServiceIntegration> _logger;
        private readonly DecisionServiceClient _decisionClient;
        private readonly DecisionServiceLauncher _launcher;
        private readonly DecisionServiceIntegrationOptions _options;
        private readonly Dictionary<string, ActiveDecision> _activeDecisions = new();
        private readonly object _lockObject = new object();

        public DecisionServiceIntegration(
            ILogger<DecisionServiceIntegration> logger,
            DecisionServiceClient decisionClient,
            DecisionServiceLauncher launcher,
            IOptions<DecisionServiceIntegrationOptions> options)
        {
            _logger = logger;
            _decisionClient = decisionClient;
            _launcher = launcher;
            _options = options.Value;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            if (!_options.Enabled)
            {
                _logger.LogInformation("🔧 Decision Service integration disabled");
                return;
            }

            try
            {
                _logger.LogInformation("🧠 Starting Decision Service integration...");

                // Wait for launcher to be ready
                await WaitForServiceReadyAsync(stoppingToken);

                // Verify connectivity
                var health = await _decisionClient.GetHealthAsync(stoppingToken);
                _logger.LogInformation("🩺 Decision Service health: {Status}, Regime: {Regime}, Daily P&L: ${DailyPnl:F2}", 
                    health.Status, health.Regime, health.DailyPnl);

                // Start periodic health monitoring
                await MonitorServiceAsync(stoppingToken);
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("🛑 Decision Service integration stopping...");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error in Decision Service integration");
            }
        }

        /// <summary>
        /// Process a new market bar through the Decision Service
        /// </summary>
        public async Task<bool> ProcessNewBarAsync(MarketBar bar, CancellationToken cancellationToken = default)
        {
            if (!IsReady()) return false;

            try
            {
                var request = new TickRequest
                {
                    Ts = bar.Timestamp.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                    Symbol = bar.Symbol,
                    O = bar.Open,
                    H = bar.High,
                    L = bar.Low,
                    C = bar.Close,
                    V = bar.Volume,
                    BidSize = bar.BidSize,
                    AskSize = bar.AskSize,
                    LastTradeDir = bar.LastTradeDirection,
                    Session = bar.Session
                };

                var response = await _decisionClient.OnNewBarAsync(request, cancellationToken);
                
                if (response.Status == "ok")
                {
                    _logger.LogDebug("📊 Bar processed: {Symbol} - Regime: {Regime}", bar.Symbol, response.Regime);
                    return true;
                }
                else
                {
                    _logger.LogWarning("⚠️ Bar processing failed: {Message}", response.Message);
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error processing bar for {Symbol}", bar.Symbol);
                return false;
            }
        }

        /// <summary>
        /// Process a strategy signal through the Decision Service ML/RL brain
        /// </summary>
        public async Task<TradingDecision> ProcessSignalAsync(StrategySignal signal, CancellationToken cancellationToken = default)
        {
            if (!IsReady())
            {
                return new TradingDecision
                {
                    ShouldTrade = false,
                    Reason = "Decision Service not ready",
                    DecisionId = Guid.NewGuid().ToString()
                };
            }

            try
            {
                var request = new SignalRequest
                {
                    Symbol = signal.Symbol,
                    StrategyId = signal.StrategyId,
                    Side = signal.Side.ToString(),
                    SignalId = signal.SignalId,
                    Hints = new Dictionary<string, object>
                    {
                        ["stopPoints"] = signal.StopPoints
                    },
                    Cloud = new CloudData
                    {
                        P = signal.Confidence,
                        SourceModelId = signal.ModelId ?? "default",
                        LatencyMs = signal.LatencyMs
                    }
                };

                var response = await _decisionClient.OnSignalAsync(request, cancellationToken);

                var decision = new TradingDecision
                {
                    ShouldTrade = response.Gate,
                    Reason = response.Reason,
                    DecisionId = response.DecisionId,
                    Symbol = signal.Symbol,
                    StrategyId = signal.StrategyId,
                    Side = signal.Side,
                    Regime = response.Regime,
                    Confidence = response.PFinal,
                    UCBScore = response.Ucb,
                    ProposedSize = response.ProposedContracts,
                    FinalSize = response.FinalContracts,
                    StopPoints = response.Risk.StopPoints,
                    PointValue = response.Risk.PointValue,
                    ManagementPlan = new TradeManagementPlan
                    {
                        TakePartialAtR = response.ManagementPlan.Tp1AtR,
                        PartialPercent = response.ManagementPlan.Tp1Pct,
                        MoveStopToBE = response.ManagementPlan.MoveStopToBEOnTp1,
                        AllowedActions = response.ManagementPlan.AllowedActions,
                        TrailATRMultiplier = response.ManagementPlan.TrailATRMultiplier
                    },
                    LatencyMs = response.LatencyMs,
                    DegradedMode = response.DegradedMode
                };

                // Store active decision for lifecycle tracking
                if (response.Gate)
                {
                    lock (_lockObject)
                    {
                        _activeDecisions[response.DecisionId] = new ActiveDecision
                        {
                            DecisionId = response.DecisionId,
                            Symbol = signal.Symbol,
                            StrategyId = signal.StrategyId,
                            Side = signal.Side,
                            SignalTime = DateTime.UtcNow,
                            Status = "pending"
                        };
                    }
                }

                _logger.LogInformation("🎯 [SIGNAL] {Strategy} {Side} {Symbol}: Gate={Gate}, Size={Size}, P={Confidence:F3}, UCB={UCB:F2}", 
                    signal.StrategyId, signal.Side, signal.Symbol, response.Gate, response.FinalContracts, response.PFinal, response.Ucb);

                return decision;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error processing signal for {Symbol} {Strategy}", signal.Symbol, signal.StrategyId);
                return new TradingDecision
                {
                    ShouldTrade = false,
                    Reason = $"Processing error: {ex.Message}",
                    DecisionId = Guid.NewGuid().ToString()
                };
            }
        }

        /// <summary>
        /// Notify Decision Service of order fill
        /// </summary>
        public async Task<bool> NotifyOrderFillAsync(OrderFill fill, CancellationToken cancellationToken = default)
        {
            if (!IsReady()) return false;

            try
            {
                var request = new FillRequest
                {
                    DecisionId = fill.DecisionId,
                    Symbol = fill.Symbol,
                    StrategyId = fill.StrategyId,
                    Side = fill.Side.ToString(),
                    EntryTs = fill.FillTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                    EntryPrice = fill.FillPrice,
                    Contracts = fill.Quantity
                };

                var response = await _decisionClient.OnOrderFillAsync(request, cancellationToken);

                if (response.Status == "ok")
                {
                    // Update active decision
                    lock (_lockObject)
                    {
                        if (_activeDecisions.TryGetValue(fill.DecisionId, out var decision))
                        {
                            decision.Status = "filled";
                            decision.FillTime = fill.FillTime;
                            decision.FillPrice = fill.FillPrice;
                            decision.Quantity = fill.Quantity;
                        }
                    }

                    _logger.LogInformation("📈 [FILL] {DecisionId}: {Quantity} {Symbol} @ ${Price:F2}", 
                        fill.DecisionId, fill.Quantity, fill.Symbol, fill.FillPrice);
                    return true;
                }
                else
                {
                    _logger.LogWarning("⚠️ Fill notification failed: {Message}", response.Message);
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error notifying fill for {DecisionId}", fill.DecisionId);
                return false;
            }
        }

        /// <summary>
        /// Notify Decision Service of trade close for online learning
        /// </summary>
        public async Task<bool> NotifyTradeCloseAsync(TradeClose close, CancellationToken cancellationToken = default)
        {
            if (!IsReady()) return false;

            try
            {
                var request = new CloseRequest
                {
                    DecisionId = close.DecisionId,
                    ExitTs = close.CloseTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"),
                    ExitPrice = close.ExitPrice,
                    FinalContracts = close.RemainingQuantity
                };

                var response = await _decisionClient.OnTradeCloseAsync(request, cancellationToken);

                if (response.Status == "ok")
                {
                    // Update and remove active decision
                    lock (_lockObject)
                    {
                        if (_activeDecisions.TryGetValue(close.DecisionId, out var decision))
                        {
                            decision.Status = "closed";
                            decision.CloseTime = close.CloseTime;
                            decision.ExitPrice = close.ExitPrice;
                            decision.PnL = response.Pnl;
                        }
                    }

                    _logger.LogInformation("💰 [CLOSE] {DecisionId}: P&L=${PnL:F2}, Daily=${DailyPnL:F2}", 
                        close.DecisionId, response.Pnl, response.DailyPnl);
                    return true;
                }
                else
                {
                    _logger.LogWarning("⚠️ Close notification failed: {Message}", response.Message);
                    return false;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error notifying close for {DecisionId}", close.DecisionId);
                return false;
            }
        }

        /// <summary>
        /// Get current Decision Service statistics
        /// </summary>
        public async Task<DecisionServiceStats> GetStatsAsync(CancellationToken cancellationToken = default)
        {
            if (!IsReady())
            {
                return new DecisionServiceStats { IsReady = false };
            }

            try
            {
                var stats = await _decisionClient.GetStatsAsync(cancellationToken);
                
                return new DecisionServiceStats
                {
                    IsReady = true,
                    Regime = stats.Regime,
                    DailyPnL = stats.DailyPnl,
                    TotalContracts = stats.TotalContracts,
                    ActivePositions = stats.ActivePositions,
                    DegradedMode = stats.DegradedMode,
                    DecisionCount = stats.DecisionCount,
                    AvgLatencyMs = stats.AvgLatencyMs,
                    Positions = stats.Positions.ConvertAll(p => new PositionSummary
                    {
                        DecisionId = p.DecisionId,
                        Symbol = p.Symbol,
                        Side = p.Side,
                        Contracts = p.Contracts,
                        Status = p.Status,
                        PnL = p.Pnl
                    })
                };
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ Error getting Decision Service stats");
                return new DecisionServiceStats { IsReady = false, ErrorMessage = ex.Message };
            }
        }

        private async Task WaitForServiceReadyAsync(CancellationToken cancellationToken)
        {
            var timeout = TimeSpan.FromSeconds(60);
            var checkInterval = TimeSpan.FromSeconds(2);
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();

            while (stopwatch.Elapsed < timeout && !cancellationToken.IsCancellationRequested)
            {
                if (_launcher.IsServiceReady)
                {
                    try
                    {
                        var health = await _decisionClient.GetHealthAsync(cancellationToken);
                        if (health.Status == "READY" || health.Status == "DEGRADED")
                        {
                            _logger.LogInformation("✅ Decision Service integration ready");
                            return;
                        }
                    }
                    catch
                    {
                        // Continue waiting
                    }
                }

                await Task.Delay(checkInterval, cancellationToken);
            }

            throw new TimeoutException("Decision Service integration did not become ready within timeout");
        }

        private async Task MonitorServiceAsync(CancellationToken cancellationToken)
        {
            var checkInterval = TimeSpan.FromSeconds(_options.HealthCheckIntervalSeconds);

            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(checkInterval, cancellationToken);

                    var health = await _decisionClient.GetHealthAsync(cancellationToken);
                    
                    if (health.Status == "ERROR")
                    {
                        _logger.LogWarning("⚠️ Decision Service health degraded: {Message}", health.Message);
                    }
                    else if (health.DegradedMode)
                    {
                        _logger.LogWarning("⚠️ Decision Service in degraded mode - latency: {LatencyMs:F1}ms", health.AvgLatencyMs);
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "⚠️ Error during Decision Service monitoring");
                }
            }
        }

        private bool IsReady()
        {
            return _options.Enabled && _launcher.IsServiceReady;
        }
    }

    /// <summary>
    /// Configuration options for Decision Service integration
    /// </summary>
    public class DecisionServiceIntegrationOptions
    {
        public bool Enabled { get; set; } = true;
        public int HealthCheckIntervalSeconds { get; set; } = 30;
        public bool LogDecisionLines { get; set; } = true;
        public bool EnableTradeManagement { get; set; } = true;
    }

    // Supporting classes
    public class ActiveDecision
    {
        public string DecisionId { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string StrategyId { get; set; } = string.Empty;
        public TradeSide Side { get; set; }
        public DateTime SignalTime { get; set; }
        public string Status { get; set; } = string.Empty;
        public DateTime? FillTime { get; set; }
        public decimal FillPrice { get; set; }
        public int Quantity { get; set; }
        public DateTime? CloseTime { get; set; }
        public decimal ExitPrice { get; set; }
        public decimal PnL { get; set; }
    }

    public class MarketBar
    {
        public DateTime Timestamp { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public decimal Open { get; set; }
        public decimal High { get; set; }
        public decimal Low { get; set; }
        public decimal Close { get; set; }
        public long Volume { get; set; }
        public int BidSize { get; set; }
        public int AskSize { get; set; }
        public int LastTradeDirection { get; set; }
        public string Session { get; set; } = "RTH";
    }

    public class StrategySignal
    {
        public string SignalId { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string StrategyId { get; set; } = string.Empty;
        public TradeSide Side { get; set; }
        public decimal Confidence { get; set; }
        public decimal StopPoints { get; set; } = 3.0m;
        public string? ModelId { get; set; }
        public int LatencyMs { get; set; }
    }

    public class TradingDecision
    {
        public bool ShouldTrade { get; set; }
        public string Reason { get; set; } = string.Empty;
        public string DecisionId { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string StrategyId { get; set; } = string.Empty;
        public TradeSide Side { get; set; }
        public string Regime { get; set; } = string.Empty;
        public decimal Confidence { get; set; }
        public decimal UCBScore { get; set; }
        public int ProposedSize { get; set; }
        public int FinalSize { get; set; }
        public decimal StopPoints { get; set; }
        public int PointValue { get; set; }
        public TradeManagementPlan ManagementPlan { get; set; } = new();
        public double LatencyMs { get; set; }
        public bool DegradedMode { get; set; }
    }

    public class TradeManagementPlan
    {
        public decimal TakePartialAtR { get; set; }
        public decimal PartialPercent { get; set; }
        public bool MoveStopToBE { get; set; }
        public List<string> AllowedActions { get; set; } = new();
        public decimal TrailATRMultiplier { get; set; }
    }

    public class OrderFill
    {
        public string DecisionId { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string StrategyId { get; set; } = string.Empty;
        public TradeSide Side { get; set; }
        public DateTime FillTime { get; set; }
        public decimal FillPrice { get; set; }
        public int Quantity { get; set; }
    }

    public class TradeClose
    {
        public string DecisionId { get; set; } = string.Empty;
        public DateTime CloseTime { get; set; }
        public decimal ExitPrice { get; set; }
        public int RemainingQuantity { get; set; }
    }

    public class DecisionServiceStats
    {
        public bool IsReady { get; set; }
        public string Regime { get; set; } = string.Empty;
        public decimal DailyPnL { get; set; }
        public int TotalContracts { get; set; }
        public int ActivePositions { get; set; }
        public bool DegradedMode { get; set; }
        public int DecisionCount { get; set; }
        public double AvgLatencyMs { get; set; }
        public List<PositionSummary> Positions { get; set; } = new();
        public string? ErrorMessage { get; set; }
    }

    public class PositionSummary
    {
        public string DecisionId { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public string Side { get; set; } = string.Empty;
        public int Contracts { get; set; }
        public string Status { get; set; } = string.Empty;
        public decimal PnL { get; set; }
    }

    public enum TradeSide
    {
        LONG,
        SHORT
    }
}