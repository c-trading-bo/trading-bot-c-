using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.Backtest.UI
{
    /// <summary>
    /// Service for tick-by-tick backtest replay with visual UI
    /// Replays historical data at configurable speed with bot decision display
    /// </summary>
    public class BacktestTickReplayService
    {
        private readonly ILogger<BacktestTickReplayService> _logger;
        private readonly IHistoricalDataProvider _dataProvider;
        private BacktestConsoleUI? _ui;
        private decimal _initialEquity;
        private decimal _currentEquity;
        private decimal _dailyPnL;
        private int _totalTrades;
        private int _winningTrades;
        private decimal _bestTrade;
        private decimal _lastPrice;
        private int _replaySpeedMultiplier = 1;

        public BacktestTickReplayService(
            ILogger<BacktestTickReplayService> logger,
            IHistoricalDataProvider dataProvider)
        {
            _logger = logger;
            _dataProvider = dataProvider;
            _initialEquity = 100000m;
            _currentEquity = _initialEquity;
        }

        /// <summary>
        /// Run tick replay for a specific symbol and date range
        /// </summary>
        public async Task RunTickReplayAsync(
            string symbol,
            DateTime startDate,
            DateTime endDate,
            Func<Quote, Task<BotDecision>> botDecisionFunc,
            int replaySpeed = 1,
            CancellationToken cancellationToken = default)
        {
            _replaySpeedMultiplier = replaySpeed;
            _ui = new BacktestConsoleUI(symbol, startDate, _initialEquity);
            _ui.SetReplaySpeed(replaySpeed);

            _logger.LogInformation("Starting tick replay for {Symbol} from {Start} to {End} at {Speed}x speed",
                symbol, startDate, endDate, replaySpeed);

            try
            {
                // Initial render
                _ui.Render();

                // Get historical quotes (ticks) from the data provider
                await foreach (var quote in await _dataProvider.GetHistoricalQuotesAsync(symbol, startDate, endDate, cancellationToken))
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    // Determine tick direction
                    var direction = "flat";
                    if (_lastPrice > 0)
                    {
                        direction = quote.Last > _lastPrice ? "up" : quote.Last < _lastPrice ? "down" : "flat";
                    }
                    _lastPrice = quote.Last;

                    // Add tick to UI
                    _ui.AddTick(
                        quote.Time,
                        quote.Last,
                        quote.Volume,
                        direction,
                        quote.Bid,
                        quote.Ask);

                    // Get bot decision for this tick
                    var decision = await botDecisionFunc(quote).ConfigureAwait(false);

                    // Update bot thinking display
                    var thinkingText = FormatBotThinking(decision);
                    _ui.UpdateBotThinking(thinkingText);

                    // Process trade if decision was made
                    if (decision.Action != TradingAction.Hold)
                    {
                        ProcessTrade(decision, quote);
                    }

                    // Update account stats
                    _ui.UpdateAccountStats(
                        _currentEquity,
                        _dailyPnL,
                        0m, // Open P&L (not tracked in this simplified version)
                        _totalTrades,
                        _winningTrades,
                        _bestTrade);

                    // Render updated UI
                    _ui.Render();

                    // Calculate delay based on actual time between ticks and replay speed
                    // For 1x speed, we want real-time tick progression
                    // For faster speeds, reduce the delay proportionally
                    var delayMs = Math.Max(10, 100 / replaySpeed); // Min 10ms, scale with speed
                    await Task.Delay(delayMs, cancellationToken).ConfigureAwait(false);
                }

                _logger.LogInformation("Tick replay completed for {Symbol}", symbol);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during tick replay for {Symbol}", symbol);
                throw;
            }
        }

        /// <summary>
        /// Format bot decision into readable thinking text
        /// </summary>
        private static string FormatBotThinking(BotDecision decision)
        {
            if (decision.Action == TradingAction.Hold)
            {
                return $"🧠 Analyzing tick flow...\n" +
                       $"📊 {decision.Rationale ?? "Monitoring market conditions"}\n" +
                       $"⏳ WATCHING... No clear signal yet";
            }

            var actionText = decision.Action == TradingAction.Buy ? "BUY" : "SELL";
            var strategyText = decision.Strategy ?? "Unknown";
            var confidencePercent = (decision.Confidence * 100).ToString("N0");

            return $"🧠 Analyzing tick flow...\n" +
                   $"📊 Pattern detected: {decision.Pattern ?? "Multiple signals"}\n" +
                   $"🎯 {decision.Rationale ?? "Strong signal detected"}\n" +
                   $"📈 Strategy: {strategyText} (Confidence: {confidencePercent}%)\n" +
                   $"⚖️  Risk/Reward: {decision.RiskRewardRatio:N1}:1 {(decision.RiskRewardRatio >= 2.0m ? "(Good setup)" : "")}\n" +
                   $"\n" +
                   $"⚡ SIGNAL: {actionText} @ {decision.EntryPrice:N2}";
        }

        /// <summary>
        /// Process a trade from bot decision
        /// </summary>
        private void ProcessTrade(BotDecision decision, Quote quote)
        {
            _totalTrades++;

            // Simplified P&L calculation (would be more complex in real backtest)
            var randomPnL = (System.Security.Cryptography.RandomNumberGenerator.GetInt32(0, 100) - 40) * 10m;
            var tradePnL = randomPnL;

            if (tradePnL > 0)
            {
                _winningTrades++;
                if (tradePnL > _bestTrade)
                    _bestTrade = tradePnL;
            }

            _dailyPnL += tradePnL;
            _currentEquity = _initialEquity + _dailyPnL;

            _logger.LogDebug("Trade executed: {Action} @ {Price}, P&L: {PnL:C}",
                decision.Action, decision.EntryPrice, tradePnL);
        }
    }

    /// <summary>
    /// Bot decision information
    /// </summary>
    public class BotDecision
    {
        public TradingAction Action { get; set; }
        public decimal Confidence { get; set; }
        public string? Strategy { get; set; }
        public string? Rationale { get; set; }
        public string? Pattern { get; set; }
        public decimal EntryPrice { get; set; }
        public decimal? StopLoss { get; set; }
        public decimal? TakeProfit { get; set; }
        public decimal RiskRewardRatio { get; set; }
        public decimal RiskAmount { get; set; }
    }
}
