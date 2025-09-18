using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using BotCore.Configuration;
using BotCore.Models;

namespace BotCore.Services
{
    /// <summary>
    /// Enhanced backtest service with market friction simulation
    /// Implements realistic slippage, latency, and commission modeling for production-ready backtesting
    /// </summary>
    public interface IEnhancedBacktestService
    {
        Task<BacktestResult> RunEnhancedBacktestAsync(BacktestRequest request, CancellationToken cancellationToken = default);
        Task<ExecutionResult> SimulateOrderExecutionAsync(OrderRequest order, MarketConditions conditions);
        decimal CalculateSlippage(string symbol, decimal price, int quantity, MarketConditions conditions);
        int CalculateExecutionLatency(OrderRequest order, MarketConditions conditions);
        decimal CalculateCommissions(string symbol, int quantity, bool isEntry = true);
    }

    /// <summary>
    /// Enhanced backtest service implementation with comprehensive market friction modeling
    /// </summary>
    public class EnhancedBacktestService : IEnhancedBacktestService
    {
        private readonly ILogger<EnhancedBacktestService> _logger;
        private readonly BacktestEnhancementConfiguration _config;
        private readonly Random _random;

        public EnhancedBacktestService(
            ILogger<EnhancedBacktestService> logger,
            IOptions<BacktestEnhancementConfiguration> config)
        {
            _logger = logger;
            _config = config.Value;
            _random = new Random();
        }

        /// <summary>
        /// Run enhanced backtest with market friction simulation
        /// </summary>
        public async Task<BacktestResult> RunEnhancedBacktestAsync(BacktestRequest request, CancellationToken cancellationToken = default)
        {
            _logger.LogInformation("[ENHANCED-BACKTEST] Starting enhanced backtest for {Strategy} from {StartDate} to {EndDate}",
                request.StrategyName, request.StartDate, request.EndDate);

            var result = new BacktestResult
            {
                StrategyName = request.StrategyName,
                StartDate = request.StartDate,
                EndDate = request.EndDate,
                Trades = new List<Trade>(),
                TotalRealisticCommissions = 0m,
                TotalSlippage = 0m,
                AverageLatencyMs = 0,
                MarketFrictionEnabled = _config.EnableMarketFriction
            };

            try
            {
                // Process each signal in the backtest
                foreach (var signal in request.Signals)
                {
                    if (cancellationToken.IsCancellationRequested)
                        break;

                    var marketConditions = CreateMarketConditions(signal.Timestamp, signal.Symbol);
                    var trade = await ProcessSignalWithFrictionAsync(signal, marketConditions).ConfigureAwait(false);
                    
                    if (trade != null)
                    {
                        result.Trades.Add(trade);
                        result.TotalRealisticCommissions += trade.Commission;
                        result.TotalSlippage += Math.Abs(trade.Slippage);
                        result.AverageLatencyMs = (result.AverageLatencyMs * (result.Trades.Count - 1) + trade.ExecutionLatencyMs) / result.Trades.Count;
                    }
                }

                // Calculate enhanced performance metrics
                CalculateEnhancedMetrics(result);

                _logger.LogInformation("[ENHANCED-BACKTEST] Completed backtest for {Strategy}: {TradeCount} trades, {Commission:C} commission, {Slippage:F4} slippage, {Latency:F1}ms avg latency",
                    request.StrategyName, result.Trades.Count, result.TotalRealisticCommissions, result.TotalSlippage, result.AverageLatencyMs);

                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ENHANCED-BACKTEST] Error running enhanced backtest for {Strategy}", request.StrategyName);
                throw;
            }
        }

        /// <summary>
        /// Simulate realistic order execution with market friction
        /// </summary>
        public async Task<ExecutionResult> SimulateOrderExecutionAsync(OrderRequest order, MarketConditions conditions)
        {
            if (!_config.EnableMarketFriction)
            {
                // Return simplified execution without friction
                return new ExecutionResult
                {
                    ExecutedPrice = order.Price,
                    ExecutedQuantity = order.Quantity,
                    ExecutionLatencyMs = 0,
                    Slippage = 0m,
                    Commission = 0m,
                    Success = true
                };
            }

            // Calculate realistic slippage
            var slippage = CalculateSlippage(order.Symbol, order.Price, order.Quantity, conditions);
            
            // Calculate execution latency
            var latency = CalculateExecutionLatency(order, conditions);
            
            // Calculate commissions
            var commission = CalculateCommissions(order.Symbol, order.Quantity, order.IsEntry);

            // Simulate latency delay
            await Task.Delay(Math.Min(latency, 100)).ConfigureAwait(false); // Cap simulation delay at 100ms

            // Determine execution price with slippage
            var isLong = order.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase);
            var slippageDirection = isLong ? 1 : -1; // Slippage always goes against the trader
            var executedPrice = order.Price + (slippage * slippageDirection);

            // Round to tick size (ES/NQ = 0.25 ticks)
            executedPrice = Px.RoundToTick(executedPrice);

            var result = new ExecutionResult
            {
                ExecutedPrice = executedPrice,
                ExecutedQuantity = order.Quantity,
                ExecutionLatencyMs = latency,
                Slippage = slippage * slippageDirection,
                Commission = commission,
                Success = true,
                MarketConditions = conditions
            };

            _logger.LogTrace("[EXECUTION-SIM] {Symbol} {Side} {Qty}@{OrigPrice} → {ExecPrice} (slip:{Slip:F4}, lat:{Lat}ms, comm:{Comm:C})",
                order.Symbol, order.Side, order.Quantity, order.Price, executedPrice, result.Slippage, latency, commission);

            return result;
        }

        /// <summary>
        /// Calculate realistic slippage based on market conditions
        /// </summary>
        public decimal CalculateSlippage(string symbol, decimal price, int quantity, MarketConditions conditions)
        {
            if (!_config.EnableMarketFriction)
                return 0m;

            var slippageConfig = _config.SlippageConfig;
            
            // Base slippage for the symbol
            var baseSlippageBps = slippageConfig.CalculateSlippage(symbol, conditions.VolatilityScore, conditions.LiquidityScore);
            
            // Convert basis points to price
            var slippageAmount = price * (decimal)(baseSlippageBps / 10000.0);
            
            // Add quantity impact (larger orders have more slippage)
            var quantityMultiplier = 1.0 + Math.Log(1 + quantity) * 0.1;
            slippageAmount *= (decimal)quantityMultiplier;
            
            // Add random variance (±20% of calculated slippage)
            var variance = 1.0 + (_random.NextDouble() - 0.5) * 0.4;
            slippageAmount *= (decimal)variance;
            
            // Ensure minimum tick size
            var tickSize = GetTickSize(symbol);
            slippageAmount = Math.Max(slippageAmount, tickSize / 4); // Minimum quarter tick

            return slippageAmount;
        }

        /// <summary>
        /// Calculate execution latency based on order and market conditions
        /// </summary>
        public int CalculateExecutionLatency(OrderRequest order, MarketConditions conditions)
        {
            if (!_config.EnableMarketFriction)
                return 0;

            var latencyConfig = _config.LatencyConfig;
            var baseLatency = latencyConfig.CalculateLatency(_random);
            
            // Market stress multiplier
            var stressMultiplier = 1.0 + (1.0 - conditions.LiquidityScore) * 0.5;
            
            // Quantity impact (larger orders take longer)
            var quantityMultiplier = 1.0 + Math.Log(1 + order.Quantity) * 0.05;
            
            var totalLatency = (int)(baseLatency * stressMultiplier * quantityMultiplier);
            return Math.Min(totalLatency, latencyConfig.MaxLatencyMs);
        }

        /// <summary>
        /// Calculate realistic commissions
        /// </summary>
        public decimal CalculateCommissions(string symbol, int quantity, bool isEntry = true)
        {
            if (!_config.EnableMarketFriction)
                return 0m;

            return _config.CommissionConfig.CalculateCommission(symbol, quantity, isEntry);
        }

        /// <summary>
        /// Process a trading signal with market friction simulation
        /// </summary>
        private async Task<Trade?> ProcessSignalWithFrictionAsync(SignalEvent signal, MarketConditions conditions)
        {
            try
            {
                // Create order from signal
                var entryOrder = new OrderRequest
                {
                    Symbol = signal.Symbol,
                    Side = signal.Side,
                    Quantity = signal.Quantity,
                    Price = signal.EntryPrice,
                    IsEntry = true,
                    Timestamp = signal.Timestamp
                };

                // Simulate entry execution
                var entryExecution = await SimulateOrderExecutionAsync(entryOrder, conditions).ConfigureAwait(false);
                if (!entryExecution.Success)
                    return null;

                // Calculate exit conditions (simplified for demonstration)
                var exitPrice = signal.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase) 
                    ? signal.TargetPrice // For longs, use target
                    : signal.StopPrice;  // For shorts, use stop (simplified)

                var exitOrder = new OrderRequest
                {
                    Symbol = signal.Symbol,
                    Side = signal.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase) ? "SELL" : "BUY",
                    Quantity = signal.Quantity,
                    Price = exitPrice,
                    IsEntry = false,
                    Timestamp = signal.Timestamp.AddMinutes(5) // Simplified - assume 5 minute hold
                };

                // Simulate exit execution
                var exitExecution = await SimulateOrderExecutionAsync(exitOrder, conditions).ConfigureAwait(false);
                if (!exitExecution.Success)
                    return null;

                // Create realistic trade record
                var trade = new Trade
                {
                    Symbol = signal.Symbol,
                    EntryTime = signal.Timestamp,
                    ExitTime = exitOrder.Timestamp,
                    Side = signal.Side,
                    Quantity = signal.Quantity,
                    EntryPrice = entryExecution.ExecutedPrice,
                    ExitPrice = exitExecution.ExecutedPrice,
                    Commission = entryExecution.Commission + exitExecution.Commission,
                    Slippage = entryExecution.Slippage + exitExecution.Slippage,
                    ExecutionLatencyMs = (entryExecution.ExecutionLatencyMs + exitExecution.ExecutionLatencyMs) / 2,
                    StrategyId = signal.StrategyId
                };

                // Calculate P&L including all costs
                var isLong = signal.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase);
                var priceDiff = trade.ExitPrice - trade.EntryPrice;
                var grossPnL = isLong ? priceDiff * signal.Quantity : -priceDiff * signal.Quantity;
                trade.GrossPnL = grossPnL;
                trade.NetPnL = grossPnL - trade.Commission; // Slippage already accounted for in prices

                return trade;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[SIGNAL-PROCESSING] Error processing signal for {Symbol}", signal.Symbol);
                return null;
            }
        }

        /// <summary>
        /// Create market conditions based on timestamp and symbol
        /// </summary>
        private MarketConditions CreateMarketConditions(DateTime timestamp, string symbol)
        {
            // Simplified market conditions simulation
            // In production, this would use actual historical market data
            var hour = timestamp.Hour;
            var isMarketOpen = (hour >= 9 && hour <= 16) || (hour >= 18 || hour <= 5); // US futures hours
            
            return new MarketConditions
            {
                Timestamp = timestamp,
                Symbol = symbol,
                VolatilityScore = 0.5 + _random.NextDouble() * 0.5, // 0.5 to 1.0
                LiquidityScore = isMarketOpen ? 0.8 + _random.NextDouble() * 0.2 : 0.3 + _random.NextDouble() * 0.4,
                MarketStress = _random.NextDouble() * 0.3, // 0 to 0.3
                IsMarketOpen = isMarketOpen
            };
        }

        /// <summary>
        /// Calculate enhanced performance metrics
        /// </summary>
        private void CalculateEnhancedMetrics(BacktestResult result)
        {
            if (!result.Trades.Any())
                return;

            // Basic performance metrics
            result.TotalTrades = result.Trades.Count;
            result.WinningTrades = result.Trades.Count(t => t.NetPnL > 0);
            result.LosingTrades = result.Trades.Count(t => t.NetPnL <= 0);
            result.WinRate = (double)result.WinningTrades / result.TotalTrades;
            
            result.GrossPnL = result.Trades.Sum(t => t.GrossPnL);
            result.NetPnL = result.Trades.Sum(t => t.NetPnL);
            result.FrictionCost = result.GrossPnL - result.NetPnL;
            
            // Enhanced metrics
            result.AverageWin = result.WinningTrades > 0 ? result.Trades.Where(t => t.NetPnL > 0).Average(t => t.NetPnL) : 0m;
            result.AverageLoss = result.LosingTrades > 0 ? result.Trades.Where(t => t.NetPnL <= 0).Average(t => t.NetPnL) : 0m;
            result.ProfitFactor = result.AverageLoss != 0 ? result.AverageWin / Math.Abs(result.AverageLoss) : 0m;
            
            // Market friction impact analysis
            result.SlippageImpactBps = result.TotalSlippage / Math.Max(1m, Math.Abs(result.GrossPnL)) * 10000;
            result.CommissionImpactBps = result.TotalRealisticCommissions / Math.Max(1m, Math.Abs(result.GrossPnL)) * 10000;
        }

        /// <summary>
        /// Get tick size for symbol
        /// </summary>
        private decimal GetTickSize(string symbol)
        {
            return symbol.ToUpper() switch
            {
                "ES" => 0.25m,
                "NQ" => 0.25m,
                _ => 0.01m
            };
        }
    }

    #region Supporting Models

    /// <summary>
    /// Enhanced backtest request with friction configuration
    /// </summary>
    public class BacktestRequest
    {
        public string StrategyName { get; set; } = string.Empty;
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public List<SignalEvent> Signals { get; } = new();
        public bool EnableMarketFriction { get; set; } = true;
    }

    /// <summary>
    /// Trading signal event for backtesting
    /// </summary>
    public class SignalEvent
    {
        public DateTime Timestamp { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public string Side { get; set; } = string.Empty; // BUY/SELL
        public int Quantity { get; set; }
        public decimal EntryPrice { get; set; }
        public decimal StopPrice { get; set; }
        public decimal TargetPrice { get; set; }
        public string StrategyId { get; set; } = string.Empty;
    }

    /// <summary>
    /// Order request for execution simulation
    /// </summary>
    public class OrderRequest
    {
        public string Symbol { get; set; } = string.Empty;
        public string Side { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal Price { get; set; }
        public bool IsEntry { get; set; }
        public DateTime Timestamp { get; set; }
    }

    /// <summary>
    /// Market conditions for realistic simulation
    /// </summary>
    public class MarketConditions
    {
        public DateTime Timestamp { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public double VolatilityScore { get; set; } // 0.0 to 1.0
        public double LiquidityScore { get; set; } // 0.0 to 1.0
        public double MarketStress { get; set; } // 0.0 to 1.0
        public bool IsMarketOpen { get; set; }
    }

    /// <summary>
    /// Execution result with friction details
    /// </summary>
    public class ExecutionResult
    {
        public decimal ExecutedPrice { get; set; }
        public int ExecutedQuantity { get; set; }
        public int ExecutionLatencyMs { get; set; }
        public decimal Slippage { get; set; }
        public decimal Commission { get; set; }
        public bool Success { get; set; }
        public MarketConditions? MarketConditions { get; set; }
    }

    /// <summary>
    /// Enhanced trade record with friction details
    /// </summary>
    public class Trade
    {
        public string Symbol { get; set; } = string.Empty;
        public DateTime EntryTime { get; set; }
        public DateTime ExitTime { get; set; }
        public string Side { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal EntryPrice { get; set; }
        public decimal ExitPrice { get; set; }
        public decimal GrossPnL { get; set; }
        public decimal NetPnL { get; set; }
        public decimal Commission { get; set; }
        public decimal Slippage { get; set; }
        public int ExecutionLatencyMs { get; set; }
        public string StrategyId { get; set; } = string.Empty;
    }

    /// <summary>
    /// Enhanced backtest result with friction analysis
    /// </summary>
    public class BacktestResult
    {
        public string StrategyName { get; set; } = string.Empty;
        public DateTime StartDate { get; set; }
        public DateTime EndDate { get; set; }
        public List<Trade> Trades { get; } = new();
        
        // Basic metrics
        public int TotalTrades { get; set; }
        public int WinningTrades { get; set; }
        public int LosingTrades { get; set; }
        public double WinRate { get; set; }
        public decimal GrossPnL { get; set; }
        public decimal NetPnL { get; set; }
        
        // Enhanced metrics
        public decimal AverageWin { get; set; }
        public decimal AverageLoss { get; set; }
        public decimal ProfitFactor { get; set; }
        
        // Friction analysis
        public bool MarketFrictionEnabled { get; set; }
        public decimal TotalRealisticCommissions { get; set; }
        public decimal TotalSlippage { get; set; }
        public double AverageLatencyMs { get; set; }
        public decimal FrictionCost { get; set; }
        public decimal SlippageImpactBps { get; set; }
        public decimal CommissionImpactBps { get; set; }
    }

    #endregion
}