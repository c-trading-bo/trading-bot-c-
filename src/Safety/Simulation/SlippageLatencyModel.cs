using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using System.Diagnostics;

namespace Trading.Safety.Simulation;

/// <summary>
/// Production-grade slippage and latency modeling for DRY_RUN simulation
/// Provides realistic execution modeling based on market conditions and order characteristics
/// </summary>
public interface ISlippageLatencyModel
{
    Task<ExecutionSimulation> SimulateExecutionAsync(OrderSimulationRequest request);
    Task<LatencyMetrics> GetCurrentLatencyMetricsAsync();
    Task<SlippageProfile> GetMarketSlippageProfileAsync(string symbol);
    Task UpdateMarketConditionsAsync(MarketConditionsUpdate update);
    event Action<ExecutionSimulation> OnExecutionSimulated;
}

/// <summary>
/// Order execution simulation request
/// </summary>
public class OrderSimulationRequest
{
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty; // BUY/SELL
    public decimal Quantity { get; set; }
    public decimal Price { get; set; }
    public string OrderType { get; set; } = string.Empty; // MARKET/LIMIT
    public DateTime RequestTime { get; set; } = DateTime.UtcNow;
    public decimal CurrentMarketPrice { get; set; }
    public decimal Volatility { get; set; } // Implied volatility for slippage modeling
    public decimal Volume { get; set; } // Recent volume for liquidity assessment
    public string Strategy { get; set; } = string.Empty;
    public Dictionary<string, object> Context { get; } = new();
}

/// <summary>
/// Comprehensive execution simulation result
/// </summary>
public class ExecutionSimulation
{
    public Guid SimulationId { get; set; } = Guid.NewGuid();
    public DateTime SimulationTime { get; set; } = DateTime.UtcNow;
    public string OrderId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    
    // Execution Results
    public decimal ExecutedPrice { get; set; }
    public decimal ExecutedQuantity { get; set; }
    public decimal Slippage { get; set; } // Difference from expected price
    public decimal SlippagePercent { get; set; }
    public TimeSpan ExecutionLatency { get; set; }
    public bool WasPartiallyFilled { get; set; }
    public decimal PartialFillRatio { get; set; } = 1.0m;
    
    // Market Impact
    public decimal MarketImpact { get; set; }
    public decimal LiquidityScore { get; set; } // 0-1, higher = better liquidity
    public string ExecutionQuality { get; set; } = string.Empty; // EXCELLENT/GOOD/FAIR/POOR
    
    // Simulation Details
    public string SimulationMethod { get; set; } = string.Empty;
    public Dictionary<string, object> SimulationFactors { get; } = new();
    public List<string> WarningFlags { get; } = new();
    
    // Performance Metrics
    public decimal EstimatedTransactionCost { get; set; }
    public decimal ImplementationShortfall { get; set; }
    public string MarketRegime { get; set; } = string.Empty;
}

/// <summary>
/// Latency performance metrics
/// </summary>
public class LatencyMetrics
{
    public DateTime LastUpdate { get; set; } = DateTime.UtcNow;
    public TimeSpan AverageLatency { get; set; }
    public TimeSpan MedianLatency { get; set; }
    public TimeSpan P95Latency { get; set; }
    public TimeSpan P99Latency { get; set; }
    public TimeSpan MaxLatency { get; set; }
    public int SampleCount { get; set; }
    public bool IsUnderStress { get; set; }
    public List<LatencyOutlier> RecentOutliers { get; } = new();
}

/// <summary>
/// Latency outlier for investigation
/// </summary>
public class LatencyOutlier
{
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public TimeSpan Latency { get; set; }
    public string Reason { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public Dictionary<string, object> Context { get; } = new();
}

/// <summary>
/// Market-specific slippage profile
/// </summary>
public class SlippageProfile
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime LastUpdate { get; set; } = DateTime.UtcNow;
    
    // Base slippage parameters (in basis points)
    public decimal BaseSlippageBps { get; set; } = 1.0m; // 0.01%
    public decimal VolatilityMultiplier { get; set; } = 1.5m;
    public decimal SizeImpactFactor { get; set; } = 0.5m;
    public decimal TimeOfDayMultiplier { get; set; } = 1.0m;
    
    // Market microstructure
    public decimal BidAskSpreadBps { get; set; }
    public decimal MarketDepth { get; set; }
    public decimal AverageTradeSize { get; set; }
    public decimal RecentVolume { get; set; }
    
    // Regime-specific adjustments
    public Dictionary<string, decimal> RegimeMultipliers { get; set; } = new()
    {
        ["NORMAL"] = 1.0m,
        ["VOLATILE"] = 2.0m,
        ["NEWS"] = 3.0m,
        ["ILLIQUID"] = 2.5m,
        ["CLOSE"] = 1.5m
    };
}

/// <summary>
/// Market conditions update
/// </summary>
public class MarketConditionsUpdate
{
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Symbol { get; set; } = string.Empty;
    public decimal CurrentPrice { get; set; }
    public decimal Volatility { get; set; }
    public decimal Volume { get; set; }
    public decimal BidAskSpread { get; set; }
    public string MarketRegime { get; set; } = "NORMAL";
    public bool IsNewsEvent { get; set; }
    public bool IsMarketOpen { get; set; } = true;
}

/// <summary>
/// Advanced slippage and latency modeling system
/// </summary>
public class SlippageLatencyModel : ISlippageLatencyModel, IHostedService
{
    private readonly ILogger<SlippageLatencyModel> _logger;
    private readonly Dictionary<string, SlippageProfile> _slippageProfiles = new();
    private readonly List<TimeSpan> _latencyHistory = new();
    private readonly List<LatencyOutlier> _latencyOutliers = new();
    private readonly Random _random = new();
    private readonly object _lock = new object();
    private readonly Timer _metricsUpdateTimer;
    
    // Configuration
    private const int MaxLatencyHistorySize = 1000;
    private const int MaxOutlierHistorySize = 100;
    private readonly TimeSpan _baseLatency = TimeSpan.FromMilliseconds(50); // Base execution latency
    private readonly TimeSpan _stressLatencyThreshold = TimeSpan.FromMilliseconds(200);
    
    public event Action<ExecutionSimulation>? OnExecutionSimulated;

    public SlippageLatencyModel(ILogger<SlippageLatencyModel> logger)
    {
        _logger = logger;
        
        // Initialize default slippage profiles
        InitializeDefaultSlippageProfiles();
        
        // Set up metrics update timer
        _metricsUpdateTimer = new Timer(UpdateMetrics, null, TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(1));
        
        _logger.LogInformation("[SLIPPAGE-MODEL] Slippage and latency modeling initialized");
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("[SLIPPAGE-MODEL] Slippage and latency model started");
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _metricsUpdateTimer?.Dispose();
        _logger.LogInformation("[SLIPPAGE-MODEL] Slippage and latency model stopped");
    }

    public async Task<ExecutionSimulation> SimulateExecutionAsync(OrderSimulationRequest request)
    {
        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            // Get slippage profile for symbol
            var profile = GetOrCreateSlippageProfile(request.Symbol);
            
            // Simulate execution latency
            var executionLatency = SimulateExecutionLatency(request);
            
            // Simulate slippage
            var slippageResult = SimulateSlippage(request, profile);
            
            // Simulate partial fills (for large orders in low liquidity)
            var fillResult = SimulatePartialFill(request, profile);
            
            // Calculate market impact
            var marketImpact = CalculateMarketImpact(request, profile);
            
            // Determine execution quality
            var executionQuality = DetermineExecutionQuality(slippageResult.SlippagePercent, executionLatency);
            
            var simulation = new ExecutionSimulation
            {
                OrderId = Guid.NewGuid().ToString("N")[..8],
                Symbol = request.Symbol,
                ExecutedPrice = slippageResult.ExecutedPrice,
                ExecutedQuantity = fillResult.FilledQuantity,
                Slippage = slippageResult.Slippage,
                SlippagePercent = slippageResult.SlippagePercent,
                ExecutionLatency = executionLatency,
                WasPartiallyFilled = fillResult.WasPartial,
                PartialFillRatio = fillResult.FillRatio,
                MarketImpact = marketImpact,
                LiquidityScore = profile.MarketDepth / 1000000m, // Normalize to 0-1
                ExecutionQuality = executionQuality,
                SimulationMethod = "MONTE_CARLO",
                EstimatedTransactionCost = CalculateTransactionCost(request, slippageResult.Slippage),
                ImplementationShortfall = CalculateImplementationShortfall(request, slippageResult.ExecutedPrice),
                MarketRegime = DetermineMarketRegime(request),
                SimulationFactors = new Dictionary<string, object>
                {
                    ["BaseSlippage"] = profile.BaseSlippageBps,
                    ["VolatilityAdjustment"] = request.Volatility,
                    ["SizeImpact"] = request.Quantity / profile.AverageTradeSize,
                    ["TimeOfDay"] = DateTime.UtcNow.TimeOfDay.TotalHours,
                    ["LiquidityScore"] = profile.MarketDepth
                }
            };
            
            // Add warning flags
            AddWarningFlags(simulation, request, profile);
            
            // Record latency for metrics
            RecordLatency(executionLatency, request.Symbol);
            
            stopwatch.Stop();
            
            _logger.LogInformation("[SLIPPAGE-MODEL] Simulated execution: {Symbol} {Side} {Qty} @ {Price:F4} " +
                                 "(Slippage: {Slippage:F4} bps, Latency: {Latency}ms, Quality: {Quality})",
                request.Symbol, request.Side, simulation.ExecutedQuantity, simulation.ExecutedPrice,
                simulation.SlippagePercent * 10000, executionLatency.TotalMilliseconds, executionQuality);
            
            // Notify subscribers
            OnExecutionSimulated?.Invoke(simulation);
            
            return simulation;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SLIPPAGE-MODEL] Error simulating execution for {Symbol}", request.Symbol);
            throw;
        }
    }

    public async Task<LatencyMetrics> GetCurrentLatencyMetricsAsync()
    {
        lock (_lock)
        {
            if (_latencyHistory.Count == 0)
            {
                return new LatencyMetrics();
            }
            
            var sortedLatencies = _latencyHistory.OrderBy(l => l.TotalMilliseconds).ToArray();
            
            return new LatencyMetrics
            {
                AverageLatency = TimeSpan.FromMilliseconds(sortedLatencies.Average(l => l.TotalMilliseconds)),
                MedianLatency = sortedLatencies[sortedLatencies.Length / 2],
                P95Latency = sortedLatencies[(int)(sortedLatencies.Length * 0.95)],
                P99Latency = sortedLatencies[(int)(sortedLatencies.Length * 0.99)],
                MaxLatency = sortedLatencies.Last(),
                SampleCount = _latencyHistory.Count,
                IsUnderStress = sortedLatencies.Average(l => l.TotalMilliseconds) > _stressLatencyThreshold.TotalMilliseconds,
                RecentOutliers = _latencyOutliers.TakeLast(10).ToList()
            };
        }
    }

    public async Task<SlippageProfile> GetMarketSlippageProfileAsync(string symbol)
    {
        return GetOrCreateSlippageProfile(symbol);
    }

    public async Task UpdateMarketConditionsAsync(MarketConditionsUpdate update)
    {
        try
        {
            var profile = GetOrCreateSlippageProfile(update.Symbol);
            
            lock (_lock)
            {
                profile.LastUpdate = update.Timestamp;
                profile.RecentVolume = update.Volume;
                profile.BidAskSpreadBps = (update.BidAskSpread / update.CurrentPrice) * 10000; // Convert to bps
                
                // Adjust time of day multiplier
                var hour = update.Timestamp.Hour;
                profile.TimeOfDayMultiplier = hour switch
                {
                    >= 9 and <= 11 => 1.5m,  // Market open volatility
                    >= 12 and <= 14 => 1.0m, // Mid-day stability
                    >= 15 and <= 16 => 1.3m, // Afternoon activity
                    _ => 2.0m                 // After hours/low liquidity
                };
                
                // Update market regime multiplier
                if (update.IsNewsEvent)
                {
                    profile.RegimeMultipliers["CURRENT"] = profile.RegimeMultipliers["NEWS"];
                }
                else if (update.Volatility > 0.02m) // 2% volatility threshold
                {
                    profile.RegimeMultipliers["CURRENT"] = profile.RegimeMultipliers["VOLATILE"];
                }
                else if (!update.IsMarketOpen)
                {
                    profile.RegimeMultipliers["CURRENT"] = profile.RegimeMultipliers["ILLIQUID"];
                }
                else
                {
                    profile.RegimeMultipliers["CURRENT"] = profile.RegimeMultipliers["NORMAL"];
                }
            }
            
            _logger.LogDebug("[SLIPPAGE-MODEL] Updated market conditions for {Symbol}: " +
                           "Volatility={Volatility:P2}, Regime={Regime}, BidAsk={BidAsk:F2}bps",
                update.Symbol, update.Volatility, update.MarketRegime, profile.BidAskSpreadBps);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SLIPPAGE-MODEL] Error updating market conditions for {Symbol}", update.Symbol);
        }
    }

    private TimeSpan SimulateExecutionLatency(OrderSimulationRequest request)
    {
        // Base latency with random variation
        var baseMs = _baseLatency.TotalMilliseconds;
        
        // Add market condition factors
        var marketMultiplier = request.Volatility > 0.02m ? 2.0 : 1.0; // High volatility increases latency
        var sizeMultiplier = 1.0 + (double)(request.Quantity / 1000m); // Large orders take longer
        
        // Add random component (log-normal distribution for realistic tail)
        var randomFactor = Math.Exp(_random.NextDouble() * 0.5 - 0.25); // 0.78 to 1.28 roughly
        
        var totalLatencyMs = baseMs * marketMultiplier * sizeMultiplier * randomFactor;
        
        return TimeSpan.FromMilliseconds(Math.Max(10, totalLatencyMs)); // Minimum 10ms
    }

    private (decimal ExecutedPrice, decimal Slippage, decimal SlippagePercent) SimulateSlippage(
        OrderSimulationRequest request, SlippageProfile profile)
    {
        if (request.OrderType == "LIMIT")
        {
            // Limit orders have no slippage if filled at limit price
            return (request.Price, 0m, 0m);
        }
        
        // Market order slippage calculation
        var baseSlippageBps = profile.BaseSlippageBps;
        
        // Volatility adjustment
        var volatilityAdjustment = Math.Min(request.Volatility * profile.VolatilityMultiplier * 10000, 50); // Cap at 50bps
        
        // Size impact
        var relativeSize = request.Quantity / profile.AverageTradeSize;
        var sizeImpact = Math.Sqrt((double)relativeSize) * (double)profile.SizeImpactFactor;
        
        // Time of day adjustment
        var timeAdjustment = profile.TimeOfDayMultiplier;
        
        // Market regime adjustment
        var regimeAdjustment = profile.RegimeMultipliers.GetValueOrDefault("CURRENT", 1.0m);
        
        // Random component (half-normal distribution for realistic skew)
        var randomComponent = Math.Abs(NextGaussian()) * 0.5; // 0 to ~2 standard deviations
        
        // Total slippage in basis points
        var totalSlippageBps = baseSlippageBps * timeAdjustment * regimeAdjustment + 
                              (decimal)volatilityAdjustment + 
                              (decimal)sizeImpact + 
                              (decimal)randomComponent;
        
        var slippagePercent = totalSlippageBps / 10000m;
        
        // Apply slippage based on order side
        var slippageDirection = request.Side == "BUY" ? 1m : -1m;
        var slippageAmount = request.CurrentMarketPrice * slippagePercent * slippageDirection;
        var executedPrice = request.CurrentMarketPrice + slippageAmount;
        
        return (executedPrice, slippageAmount, slippagePercent);
    }

    private (decimal FilledQuantity, bool WasPartial, decimal FillRatio) SimulatePartialFill(
        OrderSimulationRequest request, SlippageProfile profile)
    {
        // Probability of partial fill based on order size vs average market depth
        var relativeSize = request.Quantity / profile.MarketDepth;
        
        if (relativeSize < 0.1m) // Less than 10% of market depth
        {
            return (request.Quantity, false, 1.0m); // Full fill
        }
        
        // Higher chance of partial fill for large orders
        var partialFillProbability = Math.Min(0.8, (double)relativeSize);
        
        if (_random.NextDouble() < partialFillProbability)
        {
            // Simulate partial fill (50% to 90% typically)
            var fillRatio = 0.5m + (decimal)_random.NextDouble() * 0.4m;
            var filledQuantity = request.Quantity * fillRatio;
            
            return (filledQuantity, true, fillRatio);
        }
        
        return (request.Quantity, false, 1.0m);
    }

    private decimal CalculateMarketImpact(OrderSimulationRequest request, SlippageProfile profile)
    {
        // Simplified market impact model (square root law)
        var relativeSize = request.Quantity / profile.MarketDepth;
        var impact = Math.Sqrt((double)relativeSize) * 0.01; // 1% impact per unit of relative size
        
        return Math.Min((decimal)impact, 0.05m); // Cap at 5%
    }

    private string DetermineExecutionQuality(decimal slippagePercent, TimeSpan latency)
    {
        var slippageBps = Math.Abs(slippagePercent * 10000);
        var latencyMs = latency.TotalMilliseconds;
        
        if (slippageBps <= 2 && latencyMs <= 100)
            return "EXCELLENT";
        if (slippageBps <= 5 && latencyMs <= 200)
            return "GOOD";
        if (slippageBps <= 10 && latencyMs <= 500)
            return "FAIR";
        
        return "POOR";
    }

    private decimal CalculateTransactionCost(OrderSimulationRequest request, decimal slippage)
    {
        var slippageCost = Math.Abs(slippage * request.Quantity);
        var commissionCost = request.Quantity * 0.50m; // $0.50 per contract assumption
        
        return slippageCost + commissionCost;
    }

    private decimal CalculateImplementationShortfall(OrderSimulationRequest request, decimal executedPrice)
    {
        // Implementation shortfall vs arrival price
        var shortfall = Math.Abs(executedPrice - request.CurrentMarketPrice) * request.Quantity;
        return shortfall;
    }

    private string DetermineMarketRegime(OrderSimulationRequest request)
    {
        if (request.Volatility > 0.03m)
            return "HIGH_VOLATILITY";
        if (request.Volume < 1000)
            return "LOW_LIQUIDITY";
        
        var hour = DateTime.UtcNow.Hour;
        if (hour < 9 || hour > 16)
            return "EXTENDED_HOURS";
        
        return "NORMAL";
    }

    private void AddWarningFlags(ExecutionSimulation simulation, OrderSimulationRequest request, SlippageProfile profile)
    {
        if (simulation.SlippagePercent > 0.001m) // More than 10 bps
        {
            simulation.WarningFlags.Add($"HIGH_SLIPPAGE: {simulation.SlippagePercent:P3}");
        }
        
        if (simulation.ExecutionLatency.TotalMilliseconds > 200)
        {
            simulation.WarningFlags.Add($"HIGH_LATENCY: {simulation.ExecutionLatency.TotalMilliseconds:F0}ms");
        }
        
        if (simulation.WasPartiallyFilled)
        {
            simulation.WarningFlags.Add($"PARTIAL_FILL: {simulation.PartialFillRatio:P1}");
        }
        
        if (request.Quantity > profile.AverageTradeSize * 5)
        {
            simulation.WarningFlags.Add("LARGE_ORDER_SIZE");
        }
    }

    private void RecordLatency(TimeSpan latency, string symbol)
    {
        lock (_lock)
        {
            _latencyHistory.Add(latency);
            
            // Keep only recent history
            if (_latencyHistory.Count > MaxLatencyHistorySize)
            {
                _latencyHistory.RemoveAt(0);
            }
            
            // Record outliers
            if (latency > _stressLatencyThreshold)
            {
                _latencyOutliers.Add(new LatencyOutlier
                {
                    Latency = latency,
                    Symbol = symbol,
                    Reason = latency > TimeSpan.FromSeconds(1) ? "SEVERE_DELAY" : "MODERATE_DELAY"
                });
                
                if (_latencyOutliers.Count > MaxOutlierHistorySize)
                {
                    _latencyOutliers.RemoveAt(0);
                }
            }
        }
    }

    private SlippageProfile GetOrCreateSlippageProfile(string symbol)
    {
        lock (_lock)
        {
            if (!_slippageProfiles.TryGetValue(symbol, out var profile))
            {
                profile = CreateDefaultSlippageProfile(symbol);
                _slippageProfiles[symbol] = profile;
            }
            
            return profile;
        }
    }

    private void InitializeDefaultSlippageProfiles()
    {
        // ES (E-mini S&P 500)
        _slippageProfiles["ES"] = new SlippageProfile
        {
            Symbol = "ES",
            BaseSlippageBps = 0.5m,
            VolatilityMultiplier = 1.2m,
            SizeImpactFactor = 0.3m,
            BidAskSpreadBps = 1.0m,
            MarketDepth = 500000m,
            AverageTradeSize = 50m
        };
        
        // NQ (E-mini NASDAQ)
        _slippageProfiles["NQ"] = new SlippageProfile
        {
            Symbol = "NQ",
            BaseSlippageBps = 0.8m,
            VolatilityMultiplier = 1.5m,
            SizeImpactFactor = 0.4m,
            BidAskSpreadBps = 1.5m,
            MarketDepth = 300000m,
            AverageTradeSize = 30m
        };
    }

    private SlippageProfile CreateDefaultSlippageProfile(string symbol)
    {
        return new SlippageProfile
        {
            Symbol = symbol,
            BaseSlippageBps = 1.0m,
            VolatilityMultiplier = 1.5m,
            SizeImpactFactor = 0.5m,
            BidAskSpreadBps = 2.0m,
            MarketDepth = 100000m,
            AverageTradeSize = 20m
        };
    }

    private double NextGaussian()
    {
        // Box-Muller transformation for normal distribution
        var u1 = 1.0 - _random.NextDouble();
        var u2 = 1.0 - _random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }

    private void UpdateMetrics(object? state)
    {
        try
        {
            _ = Task.Run(async () =>
            {
                var metrics = await GetCurrentLatencyMetricsAsync().ConfigureAwait(false);
                
                if (metrics.IsUnderStress)
                {
                    _logger.LogWarning("[SLIPPAGE-MODEL] ⚠️ Latency stress detected: Avg={AvgLatency}ms, P95={P95Latency}ms",
                        metrics.AverageLatency.TotalMilliseconds, metrics.P95Latency.TotalMilliseconds);
                }
                
                _logger.LogDebug("[SLIPPAGE-MODEL] Latency metrics: Avg={AvgLatency}ms, P95={P95Latency}ms, Samples={Samples}",
                    metrics.AverageLatency.TotalMilliseconds, metrics.P95Latency.TotalMilliseconds, metrics.SampleCount);
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SLIPPAGE-MODEL] Error updating metrics");
        }
    }

    public void Dispose()
    {
        _metricsUpdateTimer?.Dispose();
    }
}