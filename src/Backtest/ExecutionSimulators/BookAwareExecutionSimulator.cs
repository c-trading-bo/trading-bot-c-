using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using Trading.Safety.Simulation;
using Trading.Safety.Journaling;

namespace TradingBot.Backtest.ExecutionSimulators
{
    /// <summary>
    /// Book-aware execution simulator that uses live fill logs to fit distributions
    /// for fill time, slippage per strategy/order type and feeds results back into
    /// execution regressor training dataset
    /// </summary>
    public class BookAwareExecutionSimulator : IExecutionSimulator
    {
        private readonly ILogger<BookAwareExecutionSimulator> _logger;
        private readonly ISlippageLatencyModel _slippageModel;
        private readonly ITradeJournal _tradeJournal;
        private readonly BookAwareSimulatorConfig _config;
        // Removed _random field - using System.Security.Cryptography.RandomNumberGenerator for secure randomness
        private readonly Dictionary<string, FillDistribution> _fillDistributions;
        private readonly object _distributionLock = new();

        public BookAwareExecutionSimulator(
            ILogger<BookAwareExecutionSimulator> logger,
            ISlippageLatencyModel slippageModel,
            ITradeJournal tradeJournal,
            IOptions<BookAwareSimulatorConfig> config)
        {
            _logger = logger;
            _slippageModel = slippageModel;
            _tradeJournal = tradeJournal;
            _config = config.Value;
            _fillDistributions = new Dictionary<string, FillDistribution>();
            
            // Initialize fill distributions from historical data
            _ = Task.Run(InitializeFillDistributionsAsync);
        }

        /// <summary>
        /// Simulate realistic order execution using book-aware modeling
        /// </summary>
        public async Task<FillResult?> SimulateOrderAsync(
            OrderSpec order, 
            Quote currentQuote, 
            SimState state, 
            CancellationToken cancellationToken = default)
        {
            try
            {
                var orderId = Guid.NewGuid().ToString("N")[..8];
                var strategy = ExtractStrategyFromOrder(order);
                
                // Get fill distribution for this strategy and order type
                var distributionKey = $"{strategy}_{order.Type}_{order.Symbol}";
                var fillDistribution = GetFillDistribution(distributionKey);
                
                // Use SlippageLatencyModel for advanced execution simulation
                var simulationRequest = new OrderSimulationRequest
                {
                    Symbol = order.Symbol,
                    OrderType = MapOrderType(order.Type),
                    Side = MapOrderSide(order.Side),
                    Quantity = order.Quantity,
                    Price = order.LimitPrice ?? currentQuote.Last,
                    CurrentMarketPrice = currentQuote.Last,
                    Volatility = 0.02m, // Default volatility estimate
                    Volume = currentQuote.Volume,
                    RequestTime = DateTime.UtcNow,
                    Strategy = ExtractStrategyFromOrder(order)
                };

                var executionSim = await _slippageModel.SimulateExecutionAsync(simulationRequest).ConfigureAwait(false);
                
                // Apply book-aware adjustments using fill distributions
                var bookAwareFillPrice = ApplyBookAwareAdjustments(
                    executionSim.ExpectedFillPrice,
                    fillDistribution,
                    order,
                    currentQuote);
                
                var fillTime = SimulateFillTime(fillDistribution, order.Type);
                var slippage = Math.Abs(bookAwareFillPrice - (order.LimitPrice ?? currentQuote.Last));
                
                // Validate fill price against market conditions
                if (!IsValidFillPrice(bookAwareFillPrice, order, currentQuote))
                {
                    _logger.LogWarning("[BOOK_AWARE_SIM] Order {OrderId} could not be filled - invalid price {FillPrice}", 
                        orderId, bookAwareFillPrice);
                    return null;
                }

                var fillResult = new FillResult(
                    orderId,
                    order.Quantity,
                    bookAwareFillPrice,
                    DateTime.UtcNow.Add(fillTime),
                    slippage,
                    $"BookAware_{fillDistribution.SampleCount}samples");

                // Update simulation state
                UpdateStateWithFill(state, order, bookAwareFillPrice);
                
                // Feed results back into execution regressor training dataset
                await FeedBackToTrainingDatasetAsync(order, fillResult, executionSim, cancellationToken).ConfigureAwait(false);

                _logger.LogDebug("[BOOK_AWARE_SIM] Executed order {OrderId}: {Quantity}@{FillPrice} (slippage: {Slippage:F4})",
                    orderId, order.Quantity, bookAwareFillPrice, slippage);

                return fillResult;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[BOOK_AWARE_SIM] Error simulating order execution for {Symbol}", order.Symbol);
                return null;
            }
        }

        /// <summary>
        /// Check for bracket order triggers using enhanced modeling
        /// </summary>
        public async Task<List<FillResult>> CheckBracketTriggersAsync(
            Quote currentQuote,
            SimState state,
            CancellationToken cancellationToken = default)
        {
            var fills = new List<FillResult>();

            foreach (var (stopLoss, takeProfit) in state.ActiveBrackets.ToList())
            {
                // Check stop-loss trigger
                if (CheckStopTrigger(stopLoss, currentQuote))
                {
                    var stopFill = await SimulateOrderAsync(stopLoss, currentQuote, state, cancellationToken).ConfigureAwait(false);
                    if (stopFill != null)
                    {
                        fills.Add(stopFill);
                        state.ActiveBrackets.Remove((stopLoss, takeProfit));
                    }
                }
                // Check take-profit trigger
                else if (CheckTakeProfitTrigger(takeProfit, currentQuote))
                {
                    var tpFill = await SimulateOrderAsync(takeProfit, currentQuote, state, cancellationToken).ConfigureAwait(false);
                    if (tpFill != null)
                    {
                        fills.Add(tpFill);
                        state.ActiveBrackets.Remove((stopLoss, takeProfit));
                    }
                }
            }

            return fills;
        }

        /// <summary>
        /// Update position and PnL based on market movement
        /// </summary>
        public void UpdatePositionPnL(Quote currentQuote, SimState state)
        {
            state.LastMarketPrice = currentQuote.Last;
            
            if (state.Position != 0 && state.AverageEntryPrice > 0)
            {
                state.UnrealizedPnL = (currentQuote.Last - state.AverageEntryPrice) * state.Position;
            }
        }

        /// <summary>
        /// Reset simulation state for new backtest run
        /// </summary>
        public void ResetState(SimState state)
        {
            state.Position = 0;
            state.AverageEntryPrice = 0;
            state.UnrealizedPnL = 0;
            state.RealizedPnL = 0;
            state.TotalCommissions = 0;
            state.RoundTripTrades = 0;
            state.WinningTrades = 0;
            state.LastMarketPrice = 0;
            state.ActiveBrackets.Clear();
        }

        /// <summary>
        /// Initialize fill distributions from historical trade journal data
        /// </summary>
        private async Task InitializeFillDistributionsAsync()
        {
            try
            {
                _logger.LogInformation("[BOOK_AWARE_SIM] Initializing fill distributions from historical data");
                
                var fromDate = DateTime.UtcNow.AddDays(-_config.HistoricalDataDays);
                var trades = await _tradeJournal.GetTradesAsync(fromDate, DateTime.UtcNow).ConfigureAwait(false);
                
                var fillsByKey = new Dictionary<string, List<FillEvent>>();
                
                foreach (var trade in trades)
                {
                    if (trade.Fills == null) continue;
                    
                    var strategy = ExtractStrategyFromTrade(trade);
                    foreach (var fill in trade.Fills)
                    {
                        var orderType = ExtractOrderTypeFromFill(fill);
                        var symbol = ExtractSymbolFromTrade(trade);
                        var key = $"{strategy}_{orderType}_{symbol}";
                        
                        if (!fillsByKey.ContainsKey(key))
                            fillsByKey[key] = new List<FillEvent>();
                        
                        fillsByKey[key].Add(fill);
                    }
                }

                lock (_distributionLock)
                {
                    foreach (var kvp in fillsByKey)
                    {
                        _fillDistributions[kvp.Key] = CalculateFillDistribution(kvp.Value);
                    }
                }

                _logger.LogInformation("[BOOK_AWARE_SIM] Initialized {DistributionCount} fill distributions from {TradeCount} trades",
                    _fillDistributions.Count, trades.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[BOOK_AWARE_SIM] Error initializing fill distributions");
            }
        }

        /// <summary>
        /// Get fill distribution for a specific key, creating default if not found
        /// </summary>
        private FillDistribution GetFillDistribution(string key)
        {
            lock (_distributionLock)
            {
                if (_fillDistributions.TryGetValue(key, out var distribution))
                    return distribution;
                
                // Create default distribution if no historical data available
                return CreateDefaultFillDistribution(key);
            }
        }

        /// <summary>
        /// Apply book-aware adjustments to fill price using historical distributions
        /// </summary>
        private decimal ApplyBookAwareAdjustments(
            decimal baseFillPrice,
            FillDistribution distribution,
            OrderSpec order,
            Quote currentQuote)
        {
            // Sample from slippage distribution
            var slippageBps = SampleFromDistribution(distribution.SlippageDistribution);
            var slippageAmount = baseFillPrice * (decimal)slippageBps / 10000m;
            
            // Apply directional slippage
            var adjustedPrice = order.Side == OrderSide.Buy
                ? baseFillPrice + Math.Abs(slippageAmount)
                : baseFillPrice - Math.Abs(slippageAmount);
            
            // Ensure price is within reasonable bounds
            var spread = currentQuote.Ask - currentQuote.Bid;
            var maxSlippage = spread * _config.MaxSlippageMultiplier;
            
            if (Math.Abs(adjustedPrice - baseFillPrice) > maxSlippage)
            {
                adjustedPrice = order.Side == OrderSide.Buy
                    ? baseFillPrice + maxSlippage
                    : baseFillPrice - maxSlippage;
            }

            return Math.Round(adjustedPrice, 2); // Round to tick size
        }

        /// <summary>
        /// Simulate fill time based on historical distribution
        /// </summary>
        private TimeSpan SimulateFillTime(FillDistribution distribution, OrderType orderType)
        {
            var fillTimeMs = orderType == OrderType.Market
                ? SampleFromDistribution(distribution.MarketFillTimeMs)
                : SampleFromDistribution(distribution.LimitFillTimeMs);
            
            return TimeSpan.FromMilliseconds(Math.Max(fillTimeMs, _config.MinFillTimeMs));
        }

        /// <summary>
        /// Sample from a normal distribution with mean and standard deviation
        /// </summary>
        private double SampleFromDistribution(DistributionStats stats)
        {
            // Box-Muller transform for normal distribution
            var u1 = 1.0 - System.Security.Cryptography.RandomNumberGenerator.GetInt32(0, 10000) / 10000.0;
            var u2 = 1.0 - System.Security.Cryptography.RandomNumberGenerator.GetInt32(0, 10000) / 10000.0;
            var standardNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            
            return stats.Mean + standardNormal * stats.StandardDeviation;
        }

        /// <summary>
        /// Calculate fill distribution from historical fill events
        /// </summary>
        private FillDistribution CalculateFillDistribution(List<FillEvent> fills)
        {
            if (fills.Count == 0)
                return CreateDefaultFillDistribution("empty");

            var slippages = new List<double>();
            var marketFillTimes = new List<double>();
            var limitFillTimes = new List<double>();

            foreach (var fill in fills)
            {
                // Extract slippage from metadata if available
                if (fill.Metadata.TryGetValue("slippage_bps", out var slippageObj) &&
                    double.TryParse(slippageObj.ToString(), out var slippage))
                {
                    slippages.Add(slippage);
                }

                // Extract fill time from metadata
                if (fill.Metadata.TryGetValue("fill_time_ms", out var fillTimeObj) &&
                    double.TryParse(fillTimeObj.ToString(), out var fillTime))
                {
                    var orderType = fill.Metadata.TryGetValue("order_type", out var orderTypeObj)
                        ? orderTypeObj.ToString()
                        : "Market";
                    
                    if (orderType == "Market")
                        marketFillTimes.Add(fillTime);
                    else
                        limitFillTimes.Add(fillTime);
                }
            }

            return new FillDistribution
            {
                SampleCount = fills.Count,
                SlippageDistribution = CalculateDistributionStats(slippages),
                MarketFillTimeMs = CalculateDistributionStats(marketFillTimes),
                LimitFillTimeMs = CalculateDistributionStats(limitFillTimes)
            };
        }

        /// <summary>
        /// Calculate distribution statistics from a list of values
        /// </summary>
        private DistributionStats CalculateDistributionStats(List<double> values)
        {
            if (values.Count == 0)
                return new DistributionStats { Mean = 0, StandardDeviation = 1.0 };

            var mean = values.Average();
            var variance = values.Sum(x => Math.Pow(x - mean, 2)) / values.Count;
            var stdDev = Math.Sqrt(variance);

            return new DistributionStats
            {
                Mean = mean,
                StandardDeviation = Math.Max(stdDev, 0.1) // Minimum std dev to avoid zero variance
            };
        }

        /// <summary>
        /// Create default fill distribution when no historical data is available
        /// </summary>
        private FillDistribution CreateDefaultFillDistribution(string key)
        {
            _logger.LogWarning("[BOOK_AWARE_SIM] Using default fill distribution for {Key}", key);
            
            return new FillDistribution
            {
                SampleCount = 0,
                SlippageDistribution = new DistributionStats { Mean = 0.5, StandardDeviation = 0.3 }, // 0.5 bps average
                MarketFillTimeMs = new DistributionStats { Mean = 50, StandardDeviation = 20 }, // 50ms average
                LimitFillTimeMs = new DistributionStats { Mean = 500, StandardDeviation = 200 } // 500ms average
            };
        }

        /// <summary>
        /// Feed simulation results back into execution regressor training dataset
        /// </summary>
        private async Task FeedBackToTrainingDatasetAsync(
            OrderSpec order,
            FillResult fill,
            ExecutionSimulation simulation,
            CancellationToken cancellationToken)
        {
            try
            {
                var trainingData = new ExecutionTrainingData
                {
                    Timestamp = DateTime.UtcNow,
                    Symbol = order.Symbol,
                    OrderType = order.Type.ToString(),
                    OrderSide = order.Side.ToString(),
                    Quantity = order.Quantity,
                    ExpectedPrice = simulation.ExpectedFillPrice,
                    ActualPrice = fill.FillPrice,
                    Slippage = fill.Slippage,
                    FillTime = fill.FillTime.Subtract(DateTime.UtcNow),
                    MarketConditions = new Dictionary<string, object>
                    {
                        ["liquidity_score"] = simulation.LiquidityScore,
                        ["execution_quality"] = simulation.ExecutionQuality,
                        ["market_regime"] = simulation.MarketRegime,
                        ["estimated_cost"] = simulation.EstimatedTransactionCost
                    }
                };

                // Store in training dataset directory
                var datasetPath = Path.Combine(_config.TrainingDatasetPath, "execution_regressor");
                Directory.CreateDirectory(datasetPath);
                
                var fileName = $"training_data_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}.json";
                var filePath = Path.Combine(datasetPath, fileName);
                
                var json = System.Text.Json.JsonSerializer.Serialize(trainingData, new System.Text.Json.JsonSerializerOptions
                {
                    WriteIndented = true,
                    PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
                });
                
                await File.WriteAllTextAsync(filePath, json, cancellationToken).ConfigureAwait(false);
                
                _logger.LogDebug("[BOOK_AWARE_SIM] Fed training data to dataset: {FilePath}", filePath);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[BOOK_AWARE_SIM] Error feeding data to training dataset");
            }
        }

        /// <summary>
        /// Validate that fill price is reasonable given market conditions
        /// </summary>
        private bool IsValidFillPrice(decimal fillPrice, OrderSpec order, Quote currentQuote)
        {
            if (fillPrice <= 0) return false;
            
            var spread = currentQuote.Ask - currentQuote.Bid;
            var maxDeviation = spread * _config.MaxPriceDeviationMultiplier;
            
            return order.Type switch
            {
                OrderType.Market => fillPrice >= currentQuote.Bid - maxDeviation && 
                                  fillPrice <= currentQuote.Ask + maxDeviation,
                OrderType.Limit => order.Side == OrderSide.Buy 
                    ? fillPrice <= (order.LimitPrice ?? decimal.MaxValue) + maxDeviation
                    : fillPrice >= (order.LimitPrice ?? 0) - maxDeviation,
                _ => true
            };
        }

        /// <summary>
        /// Check if stop-loss order should be triggered
        /// </summary>
        private bool CheckStopTrigger(OrderSpec stopOrder, Quote currentQuote)
        {
            if (stopOrder.Type != OrderType.Stop || stopOrder.StopPrice == null)
                return false;
                
            return stopOrder.Side == OrderSide.Sell 
                ? currentQuote.Last <= stopOrder.StopPrice
                : currentQuote.Last >= stopOrder.StopPrice;
        }

        /// <summary>
        /// Check if take-profit order should be triggered
        /// </summary>
        private bool CheckTakeProfitTrigger(OrderSpec profitOrder, Quote currentQuote)
        {
            if (profitOrder.Type != OrderType.Limit || profitOrder.LimitPrice == null)
                return false;
                
            return profitOrder.Side == OrderSide.Sell 
                ? currentQuote.Last >= profitOrder.LimitPrice
                : currentQuote.Last <= profitOrder.LimitPrice;
        }

        /// <summary>
        /// Update simulation state with fill results
        /// </summary>
        private void UpdateStateWithFill(SimState state, OrderSpec order, decimal fillPrice)
        {
            var fillQuantity = order.Side == OrderSide.Buy ? order.Quantity : -order.Quantity;
            var oldPosition = state.Position;
            var newPosition = oldPosition + fillQuantity;

            if (oldPosition == 0)
            {
                // Opening new position
                state.Position = newPosition;
                state.AverageEntryPrice = fillPrice;
            }
            else if (Math.Sign(oldPosition) == Math.Sign(fillQuantity))
            {
                // Adding to position
                var totalValue = (state.AverageEntryPrice * Math.Abs(oldPosition)) + (fillPrice * Math.Abs(fillQuantity));
                state.AverageEntryPrice = totalValue / Math.Abs(newPosition);
                state.Position = newPosition;
            }
            else
            {
                // Reducing or reversing position
                var closedQuantity = Math.Min(Math.Abs(oldPosition), Math.Abs(fillQuantity));
                var pnlPerContract = (fillPrice - state.AverageEntryPrice) * Math.Sign(oldPosition);
                var tradePnL = pnlPerContract * closedQuantity;
                state.RealizedPnL += tradePnL;
                state.Position = newPosition;

                if (Math.Abs(oldPosition) == Math.Abs(fillQuantity))
                {
                    // Position closed - track if it was a winning trade
                    state.RoundTripTrades++;
                    
                    // Track best trade
                    if (tradePnL > state.BestTrade)
                    {
                        state.BestTrade = tradePnL;
                    }
                    
                    if (tradePnL > 0)
                    {
                        state.WinningTrades++;
                    }
                    state.AverageEntryPrice = 0m;
                }
                else if (newPosition != 0 && Math.Sign(oldPosition) != Math.Sign(newPosition))
                {
                    // Position reversed
                    state.AverageEntryPrice = fillPrice;
                }
            }

            // Add commission
            state.TotalCommissions += _config.CommissionPerContract * Math.Abs(fillQuantity);
        }

        // Helper methods for mapping and extraction
        private string ExtractStrategyFromOrder(OrderSpec order) => "DefaultStrategy"; // Extract from order metadata
        private string ExtractStrategyFromTrade(TradeJournalEntry trade) => "DefaultStrategy"; // Extract from trade metadata
        private string ExtractOrderTypeFromFill(FillEvent fill) => "Market"; // Extract from fill metadata
        private string ExtractSymbolFromTrade(TradeJournalEntry trade) => trade.Decision?.Symbol ?? "ES";

        private string MapOrderType(OrderType orderType) => orderType.ToString();
        private string MapOrderSide(OrderSide orderSide) => orderSide.ToString();
    }

    /// <summary>
    /// Configuration for book-aware execution simulator
    /// </summary>
    public class BookAwareSimulatorConfig
    {
        public int HistoricalDataDays { get; set; } = 30;
        public decimal MaxSlippageMultiplier { get; set; } = 2.0m;
        public decimal MaxPriceDeviationMultiplier { get; set; } = 3.0m;
        public decimal CommissionPerContract { get; set; } = 2.50m;
        public double MinFillTimeMs { get; set; } = 10.0;
        public string TrainingDatasetPath { get; set; } = "data/training/execution";
    }

    /// <summary>
    /// Fill distribution data for a specific strategy/order type combination
    /// </summary>
    public class FillDistribution
    {
        public int SampleCount { get; set; }
        public DistributionStats SlippageDistribution { get; set; } = new();
        public DistributionStats MarketFillTimeMs { get; set; } = new();
        public DistributionStats LimitFillTimeMs { get; set; } = new();
    }

    /// <summary>
    /// Statistical distribution parameters
    /// </summary>
    public class DistributionStats
    {
        public double Mean { get; set; }
        public double StandardDeviation { get; set; }
    }

    /// <summary>
    /// Training data for execution regressor
    /// </summary>
    public class ExecutionTrainingData
    {
        public DateTime Timestamp { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public string OrderType { get; set; } = string.Empty;
        public string OrderSide { get; set; } = string.Empty;
        public decimal Quantity { get; set; }
        public decimal ExpectedPrice { get; set; }
        public decimal ActualPrice { get; set; }
        public decimal Slippage { get; set; }
        public TimeSpan FillTime { get; set; }
        public Dictionary<string, object> MarketConditions { get; set; } = new();
    }
}