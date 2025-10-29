using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using TradingBot.Backtest.UI;

namespace TradingBot.Backtest
{
    /// <summary>
    /// Backtest configuration options
    /// </summary>
    public class BacktestOptions
    {
        /// <summary>
        /// Commission per contract in dollars
        /// </summary>
        public decimal CommissionPerContract { get; set; } = 2.50m;

        /// <summary>
        /// Base slippage as percentage of spread
        /// </summary>
        public decimal BaseSlippagePercent { get; set; } = 0.5m;

        /// <summary>
        /// Initial capital for backtesting
        /// </summary>
        public decimal InitialCapital { get; set; } = 100000m;

        /// <summary>
        /// Maximum position size as percentage of capital
        /// </summary>
        public decimal MaxPositionSizePercent { get; set; } = 0.02m;

        /// <summary>
        /// Enable live tick replay UI (fancy visual mode)
        /// </summary>
        public bool EnableTickReplayUI { get; set; } = true;

        /// <summary>
        /// Tick replay speed multiplier (1 = real-time, 2 = 2x speed, etc.)
        /// </summary>
        public int ReplaySpeed { get; set; } = 1;
        
        /// <summary>
        /// Data granularity for backtest replay (Tick, Bar1m, Bar5m)
        /// Supports both tick-by-tick and bar-by-bar replay
        /// </summary>
        public string DataGranularity { get; set; } = "Tick";
    }

    /// <summary>
    /// Production-ready backtest harness service
    /// REPLACES the simulated SimulateModelTestingAsync() method with real historical data processing
    /// Uses existing trading services and infrastructure
    /// </summary>
    public class BacktestHarnessService
    {
        private readonly ILogger<BacktestHarnessService> _logger;
        private readonly BacktestOptions _options;
        private readonly IHistoricalDataProvider _dataProvider;
        private readonly IExecutionSimulator _executionSimulator;
        private readonly IMetricSink _metricSink;
        private readonly IModelRegistry _modelRegistry;
        private readonly IMLConfigurationService _mlConfigService;
        private readonly global::BotCore.Services.UnifiedDecisionRouter? _decisionRouter;

        public BacktestHarnessService(
            ILogger<BacktestHarnessService> logger,
            IOptions<BacktestOptions> options,
            IHistoricalDataProvider dataProvider,
            IExecutionSimulator executionSimulator,
            IMetricSink metricSink,
            IModelRegistry modelRegistry,
            IMLConfigurationService mlConfigService,
            global::BotCore.Services.UnifiedDecisionRouter? decisionRouter = null)
        {
            _logger = logger;
            _options = options.Value;
            _dataProvider = dataProvider;
            _executionSimulator = executionSimulator;
            _metricSink = metricSink;
            _modelRegistry = modelRegistry;
            _mlConfigService = mlConfigService ?? throw new ArgumentNullException(nameof(mlConfigService));
            _decisionRouter = decisionRouter; // Optional - uses real trading logic when available
        }

        /// <summary>
        /// Run comprehensive backtest using real historical data and live trading logic
        /// COMPLETELY REPLACES simulated SimulateModelTestingAsync() method
        /// Processes real historical data through existing trading pipeline
        /// </summary>
        /// <param name="symbol">Trading symbol</param>
        /// <param name="startDate">Backtest start date</param>
        /// <param name="endDate">Backtest end date</param>
        /// <param name="modelFamily">Model family to use</param>
        /// <param name="cancellationToken">Cancellation token</param>
        /// <returns>BacktestReport with legitimate performance metrics</returns>
        public async Task<BacktestReport> RunAsync(
            string symbol,
            DateTime startDate,
            DateTime endDate,
            string modelFamily,
            CancellationToken cancellationToken = default)
        {
            // SECURITY: Comprehensive input validation
            if (string.IsNullOrWhiteSpace(symbol))
                throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));
            
            if (string.IsNullOrWhiteSpace(modelFamily))
                throw new ArgumentException("Model family cannot be null or empty", nameof(modelFamily));

            if (startDate >= endDate)
                throw new ArgumentException("Start date must be before end date", nameof(endDate));

            if (endDate > DateTime.UtcNow.Date)
                throw new ArgumentException("End date cannot be in the future", nameof(endDate));

            var timeSpan = endDate - startDate;
            if (timeSpan.TotalDays < 1)
                throw new ArgumentException("Backtest period must be at least 1 day", nameof(endDate));

            if (timeSpan.TotalDays > 365)
                throw new ArgumentException("Backtest period cannot exceed 365 days", nameof(endDate));

            // SECURITY: Sanitize inputs
            if (!System.Text.RegularExpressions.Regex.IsMatch(symbol, @"^[A-Z0-9]+$"))
                throw new ArgumentException("Symbol must contain only uppercase letters and numbers", nameof(symbol));

            if (!System.Text.RegularExpressions.Regex.IsMatch(modelFamily, @"^[A-Za-z0-9_]+$"))
                throw new ArgumentException("Model family must contain only letters, numbers, and underscores", nameof(modelFamily));

            // Check if UI is enabled - suppress logging if so
            var uiEnabledEnv = Environment.GetEnvironmentVariable("ENABLE_BACKTEST_UI");
            var uiEnabled = !string.IsNullOrEmpty(uiEnabledEnv)
                ? (uiEnabledEnv == "1" || uiEnabledEnv.Equals("true", StringComparison.OrdinalIgnoreCase))
                : _options.EnableTickReplayUI;
            
            if (!uiEnabled)
            {
                _logger.LogInformation("Starting backtest for {Symbol} from {StartDate} to {EndDate} using {ModelFamily}",
                    symbol, startDate, endDate, modelFamily);
            }

            var report = new BacktestReport
            {
                Symbol = symbol,
                StartDate = startDate,
                EndDate = endDate,
                ModelFamily = modelFamily,
                InitialCapital = _options.InitialCapital,
                StartTime = DateTime.UtcNow
            };

            try
            {
                cancellationToken.ThrowIfCancellationRequested();

                // 1. Validate data availability
                if (!await _dataProvider.IsDataAvailableAsync(symbol, startDate, endDate, cancellationToken))
                {
                    throw new InvalidOperationException($"Historical data not available for {symbol} from {startDate} to {endDate}");
                }

                // 2. Get historical model (prevents future leakage)
                var model = await _modelRegistry.GetModelAsOfDateAsync(modelFamily, startDate, cancellationToken);
                if (model == null)
                {
                    throw new InvalidOperationException($"No historical model available for {modelFamily} as of {startDate}");
                }

                report.ModelId = model.ModelId;
                report.ModelVersion = model.Version;

                // 3. Initialize simulation state
                var simState = new SimState
                {
                    LastMarketPrice = 0m
                };
                _executionSimulator.ResetState(simState);

                // 3.5. Initialize tick replay UI if enabled
                BacktestConsoleUI? ui = null;
                decimal lastPrice = 0m;
                var tickCount = 0;
                
                if (uiEnabled)
                {
                    // Load all quotes first for interactive playback
                    var allQuotes = new List<Quote>();
                    await foreach (var quote in await _dataProvider.GetHistoricalQuotesAsync(symbol, startDate, endDate, cancellationToken))
                    {
                        allQuotes.Add(quote);
                    }
                    
                    ui = new BacktestConsoleUI(symbol, startDate, _options.InitialCapital);
                    ui.SetDateRange(startDate, endDate, allQuotes.Count, _options.DataGranularity);
                    ui.SetPlaybackState(UI.PlaybackState.Stopped);
                    ui.Render();
                    
                    // Run interactive playback
                    await RunInteractivePlaybackAsync(ui, symbol, model, allQuotes, simState, cancellationToken);
                }
                else
                {
                    _logger.LogInformation("📊 [BACKTEST] Running in silent mode (no UI)");
                    
                    // Non-interactive mode - process all quotes automatically
                    await foreach (var quote in await _dataProvider.GetHistoricalQuotesAsync(symbol, startDate, endDate, cancellationToken))
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        tickCount++;
                        
                        lastPrice = await ProcessSingleTickAsync(quote, model, simState, null, lastPrice, tickCount, cancellationToken);
                    }
                }

                // 5. Calculate final metrics from actual trades
                await CalculateFinalMetricsAsync(report, simState, cancellationToken);

                report.EndTime = DateTime.UtcNow;
                report.Success = true;

                if (!uiEnabled)
                {
                    _logger.LogInformation("Backtest completed successfully. Final PnL: {PnL:C}, Trades: {Trades}",
                        report.TotalPnL, report.TotalTrades);
                }

                return report;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Backtest failed for {Symbol}", symbol);
                report.Success = false;
                report.ErrorMessage = ex.Message;
                report.EndTime = DateTime.UtcNow;
                return report;
            }
        }

        /// <summary>
        /// Make trading decision using real UnifiedDecisionRouter
        /// Integrates with existing ML/RL models and strategy services
        /// </summary>
        private async Task<DecisionLog> MakeTradingDecisionAsync(
            Quote quote,
            ModelCard model,
            SimState simState,
            CancellationToken cancellationToken)
        {
            var decision = TradingAction.Hold;
            var confidence = (decimal)_mlConfigService.GetMinimumConfidence();
            var rationale = "Hold - no clear signal";
            decimal? stopLoss = null;
            decimal? takeProfit = null;

            // Use real UnifiedDecisionRouter if available
            if (_decisionRouter != null)
            {
                try
                {
                    // Create market context from quote data
                    var marketContext = new TradingBot.Abstractions.MarketContext
                    {
                        Symbol = quote.Symbol,
                        Price = (double)quote.Last,
                        Bid = (double)quote.Bid,
                        Ask = (double)quote.Ask,
                        Volume = (double)quote.Volume,
                        Timestamp = quote.Time
                    };

                    // Get unified trading decision from router (uses all ML/RL models)
                    var routerDecision = await _decisionRouter.RouteDecisionAsync(
                        quote.Symbol,
                        marketContext,
                        cancellationToken).ConfigureAwait(false);

                    // Convert router decision to backtest decision log
                    decision = routerDecision.Action switch
                    {
                        TradingBot.Abstractions.TradingAction.Buy => TradingAction.Buy,
                        TradingBot.Abstractions.TradingAction.Sell => TradingAction.Sell,
                        _ => TradingAction.Hold
                    };

                    confidence = routerDecision.Confidence;
                    rationale = routerDecision.Strategy + " - " + routerDecision.DecisionSource;
                    
                    // Calculate stop loss and take profit based on confidence
                    if (decision != TradingAction.Hold)
                    {
                        var riskPercent = 0.02m; // 2% risk
                        stopLoss = decision == TradingAction.Buy ? quote.Last * (1 - riskPercent) : quote.Last * (1 + riskPercent);
                        takeProfit = decision == TradingAction.Buy ? quote.Last * (1 + riskPercent * 2) : quote.Last * (1 - riskPercent * 2);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to get decision from UnifiedDecisionRouter, using fallback logic");
                    // Fall through to fallback logic below
                }
            }

            // Fallback logic if router not available or failed (for testing/development)
            if (decision == TradingAction.Hold && _decisionRouter == null)
            {
                var spread = quote.Ask - quote.Bid;
                var spreadPercent = quote.Last > 0 ? spread / quote.Last : 0m;

                // Basic momentum-based decision for testing only
                if (spreadPercent < 0.001m && quote.Volume > 1000)
                {
                    var hashCode = quote.Time.GetHashCode();
                    var signal = (double)((uint)hashCode % 10000) / 10000.0;
                    
                    if (signal > 0.6)
                    {
                        decision = TradingAction.Buy;
                        confidence = (decimal)_mlConfigService.GetAIConfidenceThreshold();
                        rationale = "Buy signal - fallback test logic";
                        stopLoss = quote.Last * 0.98m;
                        takeProfit = quote.Last * 1.02m;
                    }
                    else if (signal < 0.4)
                    {
                        decision = TradingAction.Sell;
                        confidence = (decimal)_mlConfigService.GetAIConfidenceThreshold();
                        rationale = "Sell signal - fallback test logic";
                        stopLoss = quote.Last * 1.02m;
                        takeProfit = quote.Last * 0.98m;
                    }
                }
            }

            var spreadData = quote.Ask - quote.Bid;
            return new DecisionLog(
                Timestamp: quote.Time,
                Symbol: quote.Symbol,
                Strategy: model.FamilyName,
                Decision: decision,
                Confidence: confidence,
                Rationale: rationale,
                EntryPrice: quote.Last,
                StopLoss: stopLoss,
                TakeProfit: takeProfit,
                RiskAmount: 1000m,
                MarketConditions: $"Spread: {spreadData:F4}, Volume: {quote.Volume}"
            );
        }

        /// <summary>
        /// Execute trading decision through simulation
        /// Uses realistic execution simulation instead of random fills
        /// </summary>
        private async Task ExecuteTradingDecisionAsync(
            DecisionLog decision,
            Quote quote,
            SimState simState,
            CancellationToken cancellationToken)
        {
            if (decision.Decision == TradingAction.Hold)
                return;

            // Simple position sizing - in production this would use sophisticated sizing logic
            var positionSize = CalculatePositionSize(decision.RiskAmount, quote.Last, decision.StopLoss ?? quote.Last * 0.98m);

            // Create order specification
            var orderSpec = new OrderSpec(
                Symbol: quote.Symbol,
                Type: OrderType.Market,
                Side: decision.Decision == TradingAction.Buy ? OrderSide.Buy : OrderSide.Sell,
                Quantity: positionSize,
                LimitPrice: null,
                StopPrice: null,
                TimeInForce: TimeInForce.Day,
                PlacedAt: quote.Time
            );

            // Execute through realistic simulation
            var fillResult = await _executionSimulator.SimulateOrderAsync(orderSpec, quote, simState, cancellationToken);
            
            if (fillResult != null)
            {
                await RecordFillAsync(fillResult, simState, cancellationToken);

                // Add bracket orders if specified
                if (decision.StopLoss.HasValue || decision.TakeProfit.HasValue)
                {
                    await AddBracketOrdersAsync(fillResult, decision, simState);
                }
            }
        }

        private decimal CalculatePositionSize(decimal riskAmount, decimal entryPrice, decimal stopLoss)
        {
            if (entryPrice == 0 || Math.Abs(entryPrice - stopLoss) < 0.01m)
                return 1m; // Default size

            var riskPerContract = Math.Abs(entryPrice - stopLoss);
            var position = riskAmount / riskPerContract;
            
            // Cap at reasonable size
            return Math.Min(Math.Max(position, 1m), 10m);
        }

        private MarketContext CreateMarketContext(Quote quote)
        {
            // Convert Quote to MarketContext for existing trading logic
            // This bridges the backtest data to live trading interfaces
            return new MarketContext
            {
                Symbol = quote.Symbol,
                Time = quote.Time,
                LastPrice = quote.Last,
                BidPrice = quote.Bid,
                AskPrice = quote.Ask,
                Volume = quote.Volume,
                Spread = quote.Ask - quote.Bid
            };
        }

        private async Task RecordDecisionAsync(DecisionLog decision, Quote quote, CancellationToken cancellationToken)
        {
            await _metricSink.RecordDecisionAsync(decision, cancellationToken);
        }

        private async Task RecordFillAsync(FillResult fill, SimState simState, CancellationToken cancellationToken)
        {
            var fillLog = new FillLog(
                Timestamp: fill.FillTime,
                OrderId: fill.OrderId,
                Symbol: "Unknown", // Fill.Symbol not available in FillResult
                Side: fill.FilledQuantity > 0 ? OrderSide.Buy : OrderSide.Sell,
                Quantity: Math.Abs(fill.FilledQuantity),
                FillPrice: fill.FillPrice,
                Slippage: fill.Slippage,
                Commission: _options.CommissionPerContract * Math.Abs(fill.FilledQuantity),
                FillReason: fill.Reason,
                RealizedPnL: simState.RealizedPnL,
                UnrealizedPnL: simState.UnrealizedPnL,
                TotalPnL: simState.RealizedPnL + simState.UnrealizedPnL
            );

            await _metricSink.RecordFillAsync(fillLog, cancellationToken);
        }

        private Task AddBracketOrdersAsync(FillResult fill, DecisionLog decision, SimState simState)
        {
            if (!decision.StopLoss.HasValue && !decision.TakeProfit.HasValue)
                return Task.CompletedTask;

            var stopLoss = decision.StopLoss.HasValue ? new OrderSpec(
                Symbol: decision.Symbol,
                Type: OrderType.Stop,
                Side: fill.FilledQuantity > 0 ? OrderSide.Sell : OrderSide.Buy,
                Quantity: Math.Abs(fill.FilledQuantity),
                LimitPrice: null,
                StopPrice: decision.StopLoss.Value,
                TimeInForce: TimeInForce.GTC,
                PlacedAt: fill.FillTime
            ) : null;

            var takeProfit = decision.TakeProfit.HasValue ? new OrderSpec(
                Symbol: decision.Symbol,
                Type: OrderType.Limit,
                Side: fill.FilledQuantity > 0 ? OrderSide.Sell : OrderSide.Buy,
                Quantity: Math.Abs(fill.FilledQuantity),
                LimitPrice: decision.TakeProfit.Value,
                StopPrice: null,
                TimeInForce: TimeInForce.GTC,
                PlacedAt: fill.FillTime
            ) : null;

            if (stopLoss != null && takeProfit != null)
            {
                simState.ActiveBrackets.Add((stopLoss, takeProfit));
            }
            
            return Task.CompletedTask;
        }

        private async Task CalculateFinalMetricsAsync(BacktestReport report, SimState simState, CancellationToken cancellationToken)
        {
            // Calculate metrics from actual simulated trades
            report.FinalCapital = _options.InitialCapital + simState.RealizedPnL + simState.UnrealizedPnL;
            report.TotalPnL = simState.RealizedPnL + simState.UnrealizedPnL;
            report.RealizedPnL = simState.RealizedPnL;
            report.UnrealizedPnL = simState.UnrealizedPnL;
            report.TotalCommissions = simState.TotalCommissions;
            report.TotalTrades = simState.RoundTripTrades;
            report.WinningTrades = simState.WinningTrades;
            report.TotalReturn = _options.InitialCapital != 0 ? report.TotalPnL / _options.InitialCapital : 0m;

            await _metricSink.FlushAsync(cancellationToken);
        }

        /// <summary>
        /// Run interactive playback loop with keyboard controls
        /// </summary>
        private async Task RunInteractivePlaybackAsync(
            BacktestConsoleUI ui,
            string symbol,
            ModelCard model,
            List<Quote> allQuotes,
            SimState simState,
            CancellationToken cancellationToken)
        {
            decimal lastPrice = 0m;
            var quit = false;
            
            while (!quit && !cancellationToken.IsCancellationRequested)
            {
                // Check for keyboard input (non-blocking)
                if (Console.KeyAvailable)
                {
                    var key = Console.ReadKey(true);
                    
                    switch (key.Key)
                    {
                        case ConsoleKey.Spacebar:
                            // Toggle play/pause
                            if (ui.GetPlaybackState() == UI.PlaybackState.Playing)
                            {
                                ui.SetPlaybackState(UI.PlaybackState.Paused);
                            }
                            else
                            {
                                ui.SetPlaybackState(UI.PlaybackState.Playing);
                            }
                            ui.Render();
                            break;
                            
                        case ConsoleKey.R:
                            // Rewind to start
                            ui.SetPlaybackState(UI.PlaybackState.Stopped);
                            ui.SetCurrentTickIndex(0);
                            simState.RealizedPnL = 0;
                            simState.UnrealizedPnL = 0;
                            simState.RoundTripTrades = 0;
                            simState.WinningTrades = 0;
                            ui.ClosePosition();
                            ui.Render();
                            break;
                            
                        case ConsoleKey.S:
                            // Stop
                            ui.SetPlaybackState(UI.PlaybackState.Stopped);
                            ui.SetCurrentTickIndex(0);
                            ui.Render();
                            break;
                            
                        case ConsoleKey.Add:
                        case ConsoleKey.OemPlus:
                            // Increase speed
                            ui.IncreaseSpeed();
                            ui.Render();
                            break;
                            
                        case ConsoleKey.Subtract:
                        case ConsoleKey.OemMinus:
                            // Decrease speed
                            ui.DecreaseSpeed();
                            ui.Render();
                            break;
                            
                        case ConsoleKey.Q:
                            // Quit
                            quit = true;
                            break;
                    }
                }
                
                // Process ticks if playing
                if (ui.GetPlaybackState() == UI.PlaybackState.Playing)
                {
                    var currentIndex = 0;
                    for (int i = 0; i < allQuotes.Count && ui.GetPlaybackState() == UI.PlaybackState.Playing && !quit; i++)
                    {
                        var quote = allQuotes[i];
                        currentIndex = i + 1;
                        ui.SetCurrentTickIndex(currentIndex);
                        
                        lastPrice = await ProcessSingleTickAsync(quote, model, simState, ui, lastPrice, currentIndex, cancellationToken);
                        
                        // Check for keyboard input between ticks
                        if (Console.KeyAvailable)
                        {
                            break; // Break to process keyboard input
                        }
                        
                        // Delay based on speed
                        var delayMs = Math.Max(10, 100 / ui.GetReplaySpeed());
                        await Task.Delay(delayMs, cancellationToken).ConfigureAwait(false);
                    }
                    
                    // If we finished all ticks, stop
                    if (currentIndex >= allQuotes.Count)
                    {
                        ui.SetPlaybackState(UI.PlaybackState.Stopped);
                        ui.Render();
                    }
                }
                else
                {
                    // If not playing, just wait a bit to avoid busy loop
                    await Task.Delay(100, cancellationToken).ConfigureAwait(false);
                }
            }
        }

        /// <summary>
        /// Process a single tick (quote) through the trading pipeline
        /// Returns the last price for tracking tick direction
        /// </summary>
        private async Task<decimal> ProcessSingleTickAsync(
            Quote quote,
            ModelCard model,
            SimState simState,
            BacktestConsoleUI? ui,
            decimal lastPrice,
            int tickCount,
            CancellationToken cancellationToken)
        {
            // Update position PnL with new market data
            _executionSimulator.UpdatePositionPnL(quote, simState);

            // Check for bracket order triggers (stop-loss, take-profit)
            var bracketFills = await _executionSimulator.CheckBracketTriggersAsync(quote, simState, cancellationToken);
            foreach (var fill in bracketFills)
            {
                await RecordFillAsync(fill, simState, cancellationToken);
            }

            // Make trading decision using EXISTING live trading logic
            var decision = await MakeTradingDecisionAsync(quote, model, simState, cancellationToken);
            
            // Record decision for analysis
            await RecordDecisionAsync(decision, quote, cancellationToken);

            // Update UI if enabled
            if (ui != null)
            {
                // Determine tick direction
                var direction = lastPrice == 0m ? "flat" : 
                               quote.Last > lastPrice ? "up" : 
                               quote.Last < lastPrice ? "down" : "flat";
                lastPrice = quote.Last;

                // Add tick to UI
                ui.AddTick(quote.Time, quote.Last, quote.Volume, direction, quote.Bid, quote.Ask);

                // Format bot thinking
                var thinkingText = FormatBotThinking(decision, quote);
                
                // Handle position UI updates and order execution display
                if (decision.Decision != TradingAction.Hold && !ui.HasOpenPosition())
                {
                    // Show signal and order submission in thinking
                    ui.UpdateBotThinking(thinkingText);
                    ui.Render();
                    await Task.Delay(500, cancellationToken).ConfigureAwait(false); // Brief pause to show order submission
                    
                    // Opening new position - show fill confirmation
                    var side = decision.Decision == TradingAction.Buy ? "LONG" : "SHORT";
                    var stopLoss = decision.StopLoss ?? quote.Last * 0.98m;
                    var target = decision.TakeProfit ?? quote.Last * 1.02m;
                    var fillPrice = decision.EntryPrice ?? quote.Last;
                    
                    // Add fill confirmation to thinking
                    thinkingText += $"\n✅ [FILL] FILLED @ {fillPrice:N2} (Slippage: 0 ticks)";
                    ui.UpdateBotThinking(thinkingText);
                    
                    ui.OpenPosition(side, fillPrice, stopLoss, target, 1, decision.Confidence, decision.Rationale);
                }
                else if (bracketFills.Count > 0 && ui.HasOpenPosition())
                {
                    // Position closed by bracket order - show exit message
                    var closeReason = bracketFills[0].Reason.Contains("stop", StringComparison.OrdinalIgnoreCase) ? "STOP LOSS" : "TAKE PROFIT";
                    var exitPrice = bracketFills[0].FillPrice;
                    thinkingText = $"🔔 [{closeReason}] Position closed @ {exitPrice:N2}\n" +
                                 $"📊 P&L will update in account stats";
                    ui.UpdateBotThinking(thinkingText);
                    ui.ClosePosition();
                }
                else
                {
                    // Normal update - just bot thinking
                    ui.UpdateBotThinking(thinkingText);
                }

                // Update account stats
                var winRate = simState.RoundTripTrades > 0 ? simState.WinningTrades : 0;
                var bestTrade = simState.BestTrade;
                ui.UpdateAccountStats(
                    _options.InitialCapital + simState.RealizedPnL + simState.UnrealizedPnL,
                    simState.RealizedPnL,
                    simState.UnrealizedPnL,
                    simState.RoundTripTrades,
                    winRate,
                    bestTrade);

                // Render UI every 5 ticks to reduce flicker
                if (tickCount % 5 == 0 || decision.Decision != TradingAction.Hold)
                {
                    ui.Render();
                }
            }

            // Execute decision if action required
            if (decision.Decision != TradingAction.Hold)
            {
                await ExecuteTradingDecisionAsync(decision, quote, simState, cancellationToken);
            }
            
            // Return the current price for next iteration
            return quote.Last;
        }

        /// <summary>
        /// Format bot decision into readable thinking text for UI display
        /// </summary>
        private static string FormatBotThinking(DecisionLog decision, Quote quote)
        {
            if (decision.Decision == TradingAction.Hold)
            {
                return $"🧠 Analyzing tick flow...\n" +
                       $"📊 {decision.Rationale}\n" +
                       $"📈 Price: {quote.Last:N2} | Spread: {(quote.Ask - quote.Bid):N4}\n" +
                       $"⏳ WATCHING... No clear signal yet";
            }

            var actionText = decision.Decision == TradingAction.Buy ? "BUY" : "SELL";
            var sideText = decision.Decision == TradingAction.Buy ? "LONG" : "SHORT";
            var confidencePercent = (decision.Confidence * 100m).ToString("N0");
            var riskReward = 0m;
            var stopDistance = 0m;
            var targetDistance = 0m;
            var stopDollar = 0m;
            var targetDollar = 0m;
            
            if (decision.TakeProfit.HasValue && decision.StopLoss.HasValue && decision.EntryPrice.HasValue && decision.EntryPrice.Value != decision.StopLoss.Value)
            {
                riskReward = Math.Abs((decision.TakeProfit.Value - decision.EntryPrice.Value) / (decision.EntryPrice.Value - decision.StopLoss.Value));
                stopDistance = Math.Abs(decision.EntryPrice.Value - decision.StopLoss.Value);
                targetDistance = Math.Abs(decision.TakeProfit.Value - decision.EntryPrice.Value);
                stopDollar = stopDistance * 50; // ES point value
                targetDollar = targetDistance * 50; // ES point value
            }

            var entryPriceText = decision.EntryPrice.HasValue ? decision.EntryPrice.Value.ToString("N2") : quote.Last.ToString("N2");

            // Create detailed signal message
            var signalDetails = $"🚨 [SIGNAL] BOT DECISION: ENTER {sideText} {decision.Symbol}!\n" +
                               $"├─ Entry: {entryPriceText}\n";
            
            if (decision.StopLoss.HasValue)
            {
                signalDetails += $"├─ Stop: {decision.StopLoss.Value:N2} (-{stopDistance:N1} pts = -${stopDollar:N0})\n";
            }
            
            if (decision.TakeProfit.HasValue)
            {
                signalDetails += $"├─ Target: {decision.TakeProfit.Value:N2} (+{targetDistance:N1} pts = +${targetDollar:N0})\n";
            }
            
            signalDetails += $"├─ Risk/Reward: {riskReward:N2}:1\n" +
                           $"├─ Confidence: {confidencePercent}%\n" +
                           $"└─ Reason: {decision.Rationale}\n\n" +
                           $"⚡ [ORDER] Submitting MARKET {actionText} 1 {decision.Symbol}...";

            return signalDetails;
        }
    }

    /// <summary>
    /// Backtest report with comprehensive metrics
    /// Replaces simulated BacktestResult with real trade-derived metrics
    /// </summary>
    public class BacktestReport
    {
        public string Symbol { get; set; } = "";
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public string ModelFamily { get; set; } = "";
        public string ModelId { get; set; } = "";
        public string ModelVersion { get; set; } = "";
        public decimal InitialCapital { get; set; }
        public decimal FinalCapital { get; set; }
        public decimal TotalPnL { get; set; }
        public decimal RealizedPnL { get; set; }
        public decimal UnrealizedPnL { get; set; }
        public decimal TotalCommissions { get; set; }
        public int TotalTrades { get; set; }
        public int WinningTrades { get; set; }
        public decimal TotalReturn { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public bool Success { get; set; }
        public string? ErrorMessage { get; set; }
    }

    /// <summary>
    /// Market context for trading decisions
    /// Bridges backtest data to existing trading interfaces
    /// </summary>
    public class MarketContext
    {
        public string Symbol { get; set; } = "";
        public DateTime Time { get; set; }
        public decimal LastPrice { get; set; }
        public decimal BidPrice { get; set; }
        public decimal AskPrice { get; set; }
        public int Volume { get; set; }
        public decimal Spread { get; set; }
    }
}