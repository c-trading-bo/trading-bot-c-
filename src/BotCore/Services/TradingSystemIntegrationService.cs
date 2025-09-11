using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TopstepX.Bot.Abstractions;
using TradingBot.Infrastructure.TopstepX;
using BotCore.Models;
using BotCore.Strategy;
using BotCore.Risk;

namespace TopstepX.Bot.Core.Services
{
    /// <summary>
    /// Unified Trading System Integration Service
    /// Coordinates all critical components for safe trading operations
    /// Implements missing components for production readiness
    /// </summary>
    public class TradingSystemIntegrationService : BackgroundService
    {
        private readonly ILogger<TradingSystemIntegrationService> _logger;
        private readonly IServiceProvider _serviceProvider;
        private readonly EmergencyStopSystem _emergencyStop;
        private readonly PositionTrackingSystem _positionTracker;
        private OrderFillConfirmationSystem? _orderConfirmation;
        private readonly ErrorHandlingMonitoringSystem _errorMonitoring;
        private readonly HttpClient _httpClient;
        private HubConnection? _userHubConnection;
        private HubConnection? _marketHubConnection;
        
        // Account/contract selection fields
        private string[] _chosenContracts = Array.Empty<string>();
        private readonly HashSet<string> _mktSubs = new(StringComparer.OrdinalIgnoreCase);
        
        // Trading system state
        private readonly TradingSystemConfiguration _config;
        private volatile bool _isSystemReady = false;
        private volatile bool _isTradingEnabled = false;
        
        // Market Data Cache - NEW IMPLEMENTATION
        private readonly ConcurrentDictionary<string, MarketData> _priceCache = new();
        private volatile int _barsSeen = 0;
        private DateTime _lastMarketDataUpdate = DateTime.MinValue;
        
        // Bar Data Storage for Strategy Evaluation - ENHANCED
        private readonly ConcurrentDictionary<string, List<Bar>> _barCache = new();
        private readonly RiskEngine _riskEngine = new();
        
        // Trading Loop state - NEW IMPLEMENTATION  
        private readonly Timer _tradingEvaluationTimer;
        private volatile bool _isEvaluationRunning = false;

        public bool IsSystemReady => _isSystemReady;
        public bool IsTradingEnabled => _isTradingEnabled && !_emergencyStop.IsEmergencyStop;
        public int BarsSeen => _barsSeen;

        public class TradingSystemConfiguration
        {
            public string TopstepXApiBaseUrl { get; set; } = "https://api.topstepx.com";
            public string UserHubUrl { get; set; } = "https://rtc.topstepx.com/hubs/user";
            public string MarketHubUrl { get; set; } = "https://rtc.topstepx.com/hubs/market";
            public string AccountId { get; set; } = string.Empty;
            public bool EnableDryRunMode { get; set; } = true;
            public bool EnableAutoExecution { get; set; } = false;
            public decimal MaxDailyLoss { get; set; } = -1000m;
            public decimal MaxPositionSize { get; set; } = 5m;
            public string ApiToken { get; set; } = string.Empty;
            public int TradingEvaluationIntervalSeconds { get; set; } = 30;
        }

        /// <summary>
        /// Market data structure for price cache
        /// </summary>
        public class MarketData
        {
            public string Symbol { get; set; } = string.Empty;
            public decimal BidPrice { get; set; }
            public decimal AskPrice { get; set; }
            public decimal LastPrice { get; set; }
            public long Volume { get; set; }
            public DateTime Timestamp { get; set; } = DateTime.UtcNow;
            public decimal Spread => AskPrice - BidPrice;
        }

        /// <summary>
        /// Place order request model matching the problem statement requirements
        /// </summary>
        public class PlaceOrderRequest
        {
            public string Symbol { get; set; } = string.Empty;
            public string Side { get; set; } = string.Empty; // BUY/SELL
            public decimal Quantity { get; set; }
            public decimal Price { get; set; }
            public decimal StopPrice { get; set; }
            public decimal TargetPrice { get; set; }
            public string OrderType { get; set; } = "LIMIT";
            public string TimeInForce { get; set; } = "DAY";
            public string CustomTag { get; set; } = string.Empty;
            public string AccountId { get; set; } = string.Empty;
        }

        public TradingSystemIntegrationService(
            ILogger<TradingSystemIntegrationService> logger,
            IServiceProvider serviceProvider,
            EmergencyStopSystem emergencyStop,
            PositionTrackingSystem positionTracker,
            ErrorHandlingMonitoringSystem errorMonitoring,
            HttpClient httpClient,
            TradingSystemConfiguration config)
        {
            _logger = logger;
            _serviceProvider = serviceProvider;
            _emergencyStop = emergencyStop;
            _positionTracker = positionTracker;
            _errorMonitoring = errorMonitoring;
            _httpClient = httpClient;
            _config = config;
            
            // Setup HTTP client
            _httpClient.BaseAddress = new Uri(_config.TopstepXApiBaseUrl);
            if (!string.IsNullOrEmpty(_config.ApiToken))
            {
                _httpClient.DefaultRequestHeaders.Authorization = 
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _config.ApiToken);
            }

            // Initialize trading evaluation timer
            _tradingEvaluationTimer = new Timer(
                EvaluateTradeOpportunitiesCallback,
                null,
                Timeout.Infinite,
                (int)TimeSpan.FromSeconds(_config.TradingEvaluationIntervalSeconds).TotalMilliseconds);

            // Initialize bar cache with symbols
            InitializeBarCache();
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            try
            {
                _logger.LogInformation("🚀 Trading System Integration Service starting...");
                
                // Initialize all components
                await InitializeComponentsAsync(stoppingToken);
                
                // Setup SignalR connections with timeout
                using var timeoutCts = new CancellationTokenSource(TimeSpan.FromMinutes(2));
                using var combinedCts = CancellationTokenSource.CreateLinkedTokenSource(stoppingToken, timeoutCts.Token);
                
                try
                {
                    await SetupSignalRConnectionsAsync(combinedCts.Token);
                }
                catch (OperationCanceledException) when (timeoutCts.IsCancellationRequested)
                {
                    _logger.LogWarning("⚠️ SignalR setup timed out after 2 minutes, continuing without live connections");
                }
                
                // Setup event handlers
                SetupEventHandlers();
                
                // Perform initial system checks
                await PerformSystemReadinessChecksAsync();
                
                _logger.LogInformation("✅ Trading System Integration Service ready");
                _isSystemReady = true;

                // Start trading evaluation timer if system is ready
                if (_isSystemReady)
                {
                    _tradingEvaluationTimer.Change(
                        10000, // Initial delay in ms
                        (int)TimeSpan.FromSeconds(_config.TradingEvaluationIntervalSeconds).TotalMilliseconds);
                }
                
                // Main service loop
                while (!stoppingToken.IsCancellationRequested)
                {
                    await MonitorSystemHealthAsync();
                    await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
                }
            }
            catch (OperationCanceledException)
            {
                _logger.LogInformation("🛑 Trading System Integration Service stopping...");
            }
            catch (Exception ex)
            {
                await _errorMonitoring.LogErrorAsync("TradingSystemIntegration", ex, ErrorHandlingMonitoringSystem.ErrorSeverity.Critical);
                _logger.LogCritical(ex, "❌ CRITICAL: Trading System Integration Service failed");
            }
            finally
            {
                await CleanupAsync();
            }
        }

        /// <summary>
        /// NEW IMPLEMENTATION: Market hours validation for CME Globex
        /// Validates trading hours for ES/MES contracts
        /// </summary>
        private bool IsMarketOpen()
        {
            try
            {
                // Convert to Eastern Time (handles DST automatically)
                var et = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, 
                    TimeZoneInfo.FindSystemTimeZoneById("America/New_York"));
                var dayOfWeek = et.DayOfWeek;
                var hour = et.Hour;

                // CME Globex hours: Sunday 6 PM ET - Friday 5 PM ET
                // Daily maintenance break: 5-6 PM ET Monday-Thursday

                // Saturday: markets closed
                if (dayOfWeek == DayOfWeek.Saturday)
                    return false;

                // Sunday: market opens at 6 PM ET
                if (dayOfWeek == DayOfWeek.Sunday)
                    return hour >= 18;

                // Friday: market closes at 5 PM ET
                if (dayOfWeek == DayOfWeek.Friday)
                    return hour < 17;

                // Monday-Thursday: daily maintenance break 5-6 PM ET
                if (dayOfWeek >= DayOfWeek.Monday && dayOfWeek <= DayOfWeek.Thursday)
                {
                    if (hour == 17) // 5 PM ET - maintenance break
                        return false;
                }

                // Regular trading hours (continuous except maintenance)
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error checking market hours, defaulting to closed");
                return false;
            }
        }

        /// <summary>
        /// NEW IMPLEMENTATION: Order placement with full validation and risk checks
        /// Implements the specifications from the problem statement
        /// </summary>
        public async Task<OrderResult> PlaceOrderAsync(PlaceOrderRequest request, CancellationToken cancellationToken = default)
        {
            try
            {
                // Check emergency stop first
                if (_emergencyStop.IsEmergencyStop)
                {
                    _logger.LogWarning("[ORDER] Order rejected - emergency stop active");
                    return OrderResult.Failed("Emergency stop active");
                }

                // Check kill.txt file
                if (File.Exists("kill.txt"))
                {
                    _logger.LogWarning("[ORDER] Order rejected - kill.txt detected, forcing DRY_RUN");
                    return OrderResult.Failed("Kill file detected");
                }

                // Market hours validation
                if (!IsMarketOpen())
                {
                    _logger.LogWarning("[ORDER] Order rejected - market is closed");
                    return OrderResult.Failed("Market is closed");
                }

                // Validate system readiness
                if (!_isSystemReady || _barsSeen < 10)
                {
                    _logger.LogWarning("[ORDER] Order rejected - system not ready. BarsSeen: {BarsSeen}, Required: 10", _barsSeen);
                    return OrderResult.Failed($"System not ready. BarsSeen: {_barsSeen}/10");
                }

                // Round prices to ES/MES tick size (0.25)
                var entryPrice = Px.RoundToTick(request.Price);
                var stopPrice = Px.RoundToTick(request.StopPrice);
                var targetPrice = Px.RoundToTick(request.TargetPrice);

                // Calculate R multiple and validate risk
                var isLong = request.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase);
                var rMultiple = Px.RMultiple(entryPrice, stopPrice, targetPrice, isLong);
                
                if (rMultiple <= 0)
                {
                    _logger.LogWarning("[ORDER] Order rejected - invalid risk calculation. R = {RMultiple}", rMultiple);
                    return OrderResult.Failed($"Invalid risk: R = {rMultiple:F2}");
                }

                // Generate unique custom tag
                var customTag = $"S11L-{DateTime.Now:yyyyMMdd-HHmmss}-{Guid.NewGuid().ToString("N")[..8]}";

                // Log structured order information per instructions
                _logger.LogInformation("[{Signal}] side={Side} symbol={Symbol} qty={Quantity} entry={Entry} stop={Stop} t1={Target} R~{RMultiple} tag={CustomTag}",
                    "SIG", request.Side, request.Symbol, request.Quantity, 
                    Px.F2(entryPrice), Px.F2(stopPrice), Px.F2(targetPrice), 
                    Px.F2(rMultiple), customTag);

                // Check if dry run mode
                if (_config.EnableDryRunMode || !_config.EnableAutoExecution)
                {
                    _logger.LogInformation("[ORDER] DRY_RUN mode - order simulation only");
                    return OrderResult.Success(customTag, customTag);
                }

                // Submit real order to TopstepX API
                var orderPayload = new
                {
                    accountId = request.AccountId,
                    symbol = request.Symbol,
                    quantity = request.Quantity,
                    price = entryPrice,
                    side = request.Side,
                    orderType = request.OrderType,
                    timeInForce = request.TimeInForce,
                    customTag = customTag
                };

                var json = JsonSerializer.Serialize(orderPayload);
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var response = await _httpClient.PostAsync("/api/Order/place", content, cancellationToken);

                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync(cancellationToken);
                    var orderResponse = JsonSerializer.Deserialize<JsonElement>(responseContent);
                    
                    var orderId = orderResponse.TryGetProperty("orderId", out var orderIdElement) 
                        ? orderIdElement.GetString() 
                        : customTag;

                    // Log order placement
                    _logger.LogInformation("[ORDER] account={AccountId} status=New orderId={OrderId} tag={CustomTag}",
                        request.AccountId, orderId, customTag);

                    return OrderResult.Success(orderId, customTag);
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                    _logger.LogWarning("[ORDER] API rejection: {StatusCode} - {Content}", response.StatusCode, errorContent);
                    return OrderResult.Failed($"API Error: {response.StatusCode} - {errorContent}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ORDER] Exception during order placement");
                return OrderResult.Failed($"Exception: {ex.Message}");
            }
        }

        /// <summary>
        /// ENHANCED IMPLEMENTATION: Trading strategy evaluation using AllStrategies
        /// Connects existing sophisticated strategies (S1-S14) to trading execution
        /// </summary>
        private async Task EvaluateTradeOpportunitiesAsync()
        {
            if (_isEvaluationRunning || !_isSystemReady)
                return;

            _isEvaluationRunning = true;
            try
            {
                // Check preconditions
                if (!await PerformTradingPrechecksAsync())
                    return;

                // Get symbols to evaluate (ES, MES, NQ, MNQ)
                var symbols = new[] { "ES", "MES", "NQ", "MNQ" };
                
                foreach (var symbol in symbols)
                {
                    await EvaluateSymbolStrategiesAsync(symbol);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[STRATEGY] Error during trade opportunity evaluation");
            }
            finally
            {
                _isEvaluationRunning = false;
            }
        }

        /// <summary>
        /// Evaluate all strategies for a specific symbol using AllStrategies.generate_candidates
        /// </summary>
        private async Task EvaluateSymbolStrategiesAsync(string symbol)
        {
            try
            {
                // Get market data for the symbol
                if (!_priceCache.TryGetValue(symbol, out var marketData))
                {
                    _logger.LogDebug("[STRATEGY] No market data available for {Symbol}", symbol);
                    return;
                }

                // Get bar data for the symbol
                if (!_barCache.TryGetValue(symbol, out var bars) || bars.Count < 20)
                {
                    _logger.LogDebug("[STRATEGY] Insufficient bar data for {Symbol} (need 20+, have {Count})", symbol, bars?.Count ?? 0);
                    return;
                }

                // Create environment for strategy evaluation
                var env = new Env
                {
                    Symbol = symbol,
                    atr = CalculateATR(bars),
                    volz = CalculateVolZ(bars) // Use our own implementation
                };

                // Create levels (placeholder - could be enhanced with real support/resistance levels)
                var levels = new Levels();

                // Generate strategy candidates using AllStrategies
                var candidates = AllStrategies.generate_candidates(symbol, env, levels, bars, _riskEngine);

                _logger.LogInformation("[STRATEGY] Generated {Count} candidates for {Symbol}", candidates.Count, symbol);

                // Process each candidate
                foreach (var candidate in candidates.Where(c => Math.Abs(c.qty) > 0))
                {
                    await ProcessStrategyCandidateAsync(candidate);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[STRATEGY] Error evaluating strategies for {Symbol}", symbol);
            }
        }

        /// <summary>
        /// Process a strategy candidate and place order if conditions are met
        /// </summary>
        private async Task ProcessStrategyCandidateAsync(Candidate candidate)
        {
            try
            {
                // Validate candidate
                if (!ValidateCandidate(candidate))
                    return;

                // Convert candidate to order request
                var orderRequest = new PlaceOrderRequest
                {
                    Symbol = candidate.symbol,
                    Side = candidate.side == Side.BUY ? "BUY" : "SELL",
                    Quantity = (decimal)Math.Abs(candidate.qty),
                    Price = candidate.entry,
                    StopPrice = candidate.stop,
                    TargetPrice = candidate.t1,
                    CustomTag = candidate.Tag,
                    AccountId = _config.AccountId
                };

                // Place the order
                var result = await PlaceOrderAsync(orderRequest);
                
                _logger.LogInformation("[STRATEGY] Strategy {StrategyId} signal executed: {Symbol} {Side} Qty={Qty} Entry={Entry} Stop={Stop} Target={Target} Result={Success}",
                    candidate.strategy_id, candidate.symbol, candidate.side, candidate.qty, 
                    Px.F2(candidate.entry), Px.F2(candidate.stop), Px.F2(candidate.t1), result.IsSuccess);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[STRATEGY] Error processing candidate for {StrategyId} {Symbol}", candidate.strategy_id, candidate.symbol);
            }
        }

        /// <summary>
        /// Validate strategy candidate before order placement
        /// </summary>
        private bool ValidateCandidate(Candidate candidate)
        {
            // Check if symbol is supported
            if (string.IsNullOrEmpty(candidate.symbol))
            {
                _logger.LogDebug("[VALIDATION] Invalid symbol for candidate {StrategyId}", candidate.strategy_id);
                return false;
            }

            // Check if prices are valid
            if (candidate.entry <= 0 || candidate.stop <= 0 || candidate.t1 <= 0)
            {
                _logger.LogDebug("[VALIDATION] Invalid prices for candidate {StrategyId} {Symbol}", candidate.strategy_id, candidate.symbol);
                return false;
            }

            // Calculate risk/reward ratio
            var isLong = candidate.side == Side.BUY;
            var risk = isLong ? candidate.entry - candidate.stop : candidate.stop - candidate.entry;
            var reward = isLong ? candidate.t1 - candidate.entry : candidate.entry - candidate.t1;

            if (risk <= 0)
            {
                _logger.LogDebug("[VALIDATION] Non-positive risk for candidate {StrategyId} {Symbol}", candidate.strategy_id, candidate.symbol);
                return false;
            }

            var rrRatio = reward / risk;
            if (rrRatio < 1.0m) // Minimum 1:1 risk/reward
            {
                _logger.LogDebug("[VALIDATION] Poor risk/reward ratio {RRRatio:F2} for candidate {StrategyId} {Symbol}", rrRatio, candidate.strategy_id, candidate.symbol);
                return false;
            }

            return true;
        }

        /// <summary>
        /// Calculate ATR (Average True Range) for volatility measurement
        /// </summary>
        private decimal CalculateATR(List<Bar> bars, int period = 14)
        {
            if (bars.Count < period + 1)
                return 1m; // Default ATR

            var trueRanges = new List<decimal>();
            
            for (int i = 1; i < bars.Count && trueRanges.Count < period; i++)
            {
                var current = bars[i];
                var previous = bars[i - 1];
                
                var tr1 = current.High - current.Low;
                var tr2 = Math.Abs(current.High - previous.Close);
                var tr3 = Math.Abs(current.Low - previous.Close);
                
                trueRanges.Add(Math.Max(tr1, Math.Max(tr2, tr3)));
            }
            
            return trueRanges.Any() ? trueRanges.Average() : 1m;
        }

        /// <summary>
        /// Calculate VolZ (volatility z-score) - regime proxy using recent returns
        /// </summary>
        private decimal CalculateVolZ(List<Bar> bars, int lookback = 50)
        {
            if (bars.Count < lookback + 1)
                return 0m;

            var returns = new List<decimal>(lookback);
            for (int i = bars.Count - lookback; i < bars.Count; i++)
            {
                if (i > 0 && bars[i - 1].Close > 0)
                {
                    var ret = (bars[i].Close / bars[i - 1].Close) - 1m;
                    returns.Add(ret);
                }
            }

            if (returns.Count < 2)
                return 0m;

            var mean = returns.Average();
            var variance = returns.Sum(r => (r - mean) * (r - mean)) / (returns.Count - 1);
            var stdDev = (decimal)Math.Sqrt((double)variance);
            
            if (stdDev <= 0)
                return 0m;

            var lastReturn = returns.LastOrDefault();
            return (lastReturn - mean) / stdDev;
        }

        /// <summary>
        /// Initialize bar cache with empty collections for supported symbols
        /// </summary>
        private void InitializeBarCache()
        {
            var symbols = new[] { "ES", "MES", "NQ", "MNQ" };
            foreach (var symbol in symbols)
            {
                _barCache.TryAdd(symbol, new List<Bar>());
                _logger.LogDebug("[BAR_CACHE] Initialized bar cache for {Symbol}", symbol);
            }
        }

        /// <summary>
        /// NEW IMPLEMENTATION: Perform all trading prechecks before execution
        /// </summary>
        private Task<bool> PerformTradingPrechecksAsync()
        {
            // Check emergency stop
            if (_emergencyStop.IsEmergencyStop)
            {
                _logger.LogDebug("[PRECHECK] Failed - emergency stop active");
                return Task.FromResult(false);
            }

            // Check kill.txt
            if (File.Exists("kill.txt"))
            {
                _logger.LogDebug("[PRECHECK] Failed - kill.txt detected");
                return Task.FromResult(false);
            }

            // Check bars seen
            if (_barsSeen < 10)
            {
                _logger.LogDebug("[PRECHECK] Failed - insufficient bars seen: {BarsSeen}/10", _barsSeen);
                return Task.FromResult(false);
            }

            // Check hub connections
            var userHubConnected = _userHubConnection?.State == HubConnectionState.Connected;
            var marketHubConnected = _marketHubConnection?.State == HubConnectionState.Connected;
            
            if (!userHubConnected || !marketHubConnected)
            {
                _logger.LogDebug("[PRECHECK] Failed - hubs not connected. User: {UserHub}, Market: {MarketHub}",
                    userHubConnected, marketHubConnected);
                return Task.FromResult(false);
            }

            // Check market hours
            if (!IsMarketOpen())
            {
                _logger.LogDebug("[PRECHECK] Failed - market is closed");
                return Task.FromResult(false);
            }

            return Task.FromResult(true);
        }

        /// <summary>
        /// Timer callback for trade evaluation
        /// </summary>
        private void EvaluateTradeOpportunitiesCallback(object? state)
        {
            _ = Task.Run(async () =>
            {
                try
                {
                    await EvaluateTradeOpportunitiesAsync();
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[STRATEGY] Error in timer callback");
                }
            });
        }

        /// <summary>
        /// ENHANCED: Market data event handler with price cache and bar data updates
        /// </summary>
        private Task OnMarketDataReceived(object marketDataObj)
        {
            try
            {
                // Parse market data (implementation depends on TopstepX format)
                var marketDataJson = JsonSerializer.Serialize(marketDataObj);
                var marketDataElement = JsonSerializer.Deserialize<JsonElement>(marketDataJson);

                if (marketDataElement.TryGetProperty("symbol", out var symbolElement) &&
                    marketDataElement.TryGetProperty("bid", out var bidElement) &&
                    marketDataElement.TryGetProperty("ask", out var askElement) &&
                    marketDataElement.TryGetProperty("last", out var lastElement))
                {
                    var symbol = symbolElement.GetString() ?? string.Empty;
                    var marketData = new MarketData
                    {
                        Symbol = symbol,
                        BidPrice = bidElement.GetDecimal(),
                        AskPrice = askElement.GetDecimal(),
                        LastPrice = lastElement.GetDecimal(),
                        Volume = marketDataElement.TryGetProperty("volume", out var volElement) ? volElement.GetInt64() : 0,
                        Timestamp = DateTime.UtcNow
                    };

                    // Update price cache
                    _priceCache.AddOrUpdate(symbol, marketData, (key, old) => marketData);
                    _lastMarketDataUpdate = DateTime.UtcNow;
                    
                    // Update bar cache for strategy evaluation
                    UpdateBarCache(symbol, marketData);
                    
                    // Increment bars seen counter
                    Interlocked.Increment(ref _barsSeen);

                    _logger.LogDebug("[MARKET_DATA] {Symbol}: Bid={Bid} Ask={Ask} Last={Last} BarsSeen={BarsSeen}",
                        symbol, Px.F2(marketData.BidPrice), Px.F2(marketData.AskPrice), 
                        Px.F2(marketData.LastPrice), _barsSeen);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[MARKET_DATA] Error processing market data");
            }
            
            return Task.CompletedTask;
        }

        /// <summary>
        /// Update bar cache with new market data for strategy evaluation
        /// </summary>
        private void UpdateBarCache(string symbol, MarketData marketData)
        {
            try
            {
                // Get or create bar list for symbol
                var bars = _barCache.GetOrAdd(symbol, _ => new List<Bar>());

                // Create a new bar from market data
                // Note: This is a simplified approach. In production, you might want to aggregate
                // multiple market data updates into proper time-based bars (1m, 5m, etc.)
                var currentTime = DateTime.UtcNow;
                
                // Check if we should create a new bar or update the current one
                var lastBar = bars.LastOrDefault();
                var shouldCreateNewBar = lastBar == null || 
                    currentTime.Subtract(lastBar.Start).TotalMinutes >= 1; // 1-minute bars

                if (shouldCreateNewBar)
                {
                    // Create new bar
                    var newBar = new Bar
                    {
                        Start = currentTime,
                        Ts = ((DateTimeOffset)currentTime).ToUnixTimeMilliseconds(),
                        Symbol = symbol,
                        Open = marketData.LastPrice,
                        High = marketData.LastPrice,
                        Low = marketData.LastPrice,
                        Close = marketData.LastPrice,
                        Volume = (int)marketData.Volume
                    };
                    
                    bars.Add(newBar);
                    
                    // Keep only last 200 bars for memory efficiency
                    if (bars.Count > 200)
                    {
                        bars.RemoveAt(0);
                    }
                }
                else if (lastBar != null)
                {
                    // Update current bar
                    lastBar.High = Math.Max(lastBar.High, marketData.LastPrice);
                    lastBar.Low = Math.Min(lastBar.Low, marketData.LastPrice);
                    lastBar.Close = marketData.LastPrice;
                    lastBar.Volume += (int)marketData.Volume;
                }

                _logger.LogTrace("[BAR_CACHE] Updated {Symbol}: {BarCount} bars, Latest: O={Open} H={High} L={Low} C={Close}",
                    symbol, bars.Count, 
                    lastBar?.Open ?? 0, lastBar?.High ?? 0, lastBar?.Low ?? 0, lastBar?.Close ?? 0);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[BAR_CACHE] Error updating bar cache for {Symbol}", symbol);
            }
        }

        /// <summary>
        /// ENHANCED: Fill confirmation handler with position updates
        /// </summary>
        private async Task OnFillConfirmed(object fillObj)
        {
            try
            {
                var fillJson = JsonSerializer.Serialize(fillObj);
                var fillElement = JsonSerializer.Deserialize<JsonElement>(fillJson);

                if (fillElement.TryGetProperty("orderId", out var orderIdElement) &&
                    fillElement.TryGetProperty("symbol", out var symbolElement) &&
                    fillElement.TryGetProperty("quantity", out var qtyElement) &&
                    fillElement.TryGetProperty("price", out var priceElement) &&
                    fillElement.TryGetProperty("side", out var sideElement))
                {
                    var orderId = orderIdElement.GetString() ?? string.Empty;
                    var symbol = symbolElement.GetString() ?? string.Empty;
                    var quantity = qtyElement.GetDecimal();
                    var price = priceElement.GetDecimal();
                    var side = sideElement.GetString() ?? string.Empty;

                    // Log fill confirmation per instructions
                    _logger.LogInformation("[TRADE] account={AccountId} orderId={OrderId} fillPrice={FillPrice} qty={Quantity} time={Time}",
                        _config.AccountId, orderId, Px.F2(price), quantity, DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ"));

                    // Update position tracker - fix parameter types
                    await _positionTracker.ProcessFillAsync(orderId, symbol, price, (int)quantity);
                    
                    _logger.LogInformation("✅ Position updated for {Symbol}: {Quantity} @ {Price}", symbol, quantity, Px.F2(price));
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[FILL] Error processing fill confirmation");
            }
        }

        // Rest of the existing methods would be implemented here...
        // (I'm keeping this focused on the new implementations for now)

        private Task InitializeComponentsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("🔧 Initializing trading system components...");
            
            // Initialize existing components
            _errorMonitoring.UpdateComponentHealth("ErrorMonitoring", ErrorHandlingMonitoringSystem.HealthStatus.Healthy);
            _emergencyStop.EmergencyStopTriggered += OnEmergencyStopTriggered;
            _errorMonitoring.UpdateComponentHealth("EmergencyStop", ErrorHandlingMonitoringSystem.HealthStatus.Healthy);
            
            // Initialize order confirmation system
            _orderConfirmation = _serviceProvider.GetService<OrderFillConfirmationSystem>();
            if (_orderConfirmation != null)
            {
                _errorMonitoring.UpdateComponentHealth("OrderConfirmation", ErrorHandlingMonitoringSystem.HealthStatus.Healthy);
            }
            
            _logger.LogInformation("✅ Trading system components initialized");
            return Task.CompletedTask;
        }

        private async Task SetupSignalRConnectionsAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("📡 Setting up SignalR connections...");
            
            // Setup User Hub connection
            _userHubConnection = new HubConnectionBuilder()
                .WithUrl(_config.UserHubUrl)
                .WithAutomaticReconnect()
                .Build();

            // Setup Market Hub connection  
            _marketHubConnection = new HubConnectionBuilder()
                .WithUrl(_config.MarketHubUrl)
                .WithAutomaticReconnect()
                .Build();

            await Task.WhenAll(
                _userHubConnection.StartAsync(cancellationToken),
                _marketHubConnection.StartAsync(cancellationToken)
            );

            _logger.LogInformation("✅ SignalR connections established");
        }

        private void SetupEventHandlers()
        {
            if (_userHubConnection != null)
            {
                _userHubConnection.On<object>("FillUpdate", OnFillConfirmed);
                _userHubConnection.On<object>("OrderUpdate", (order) => 
                {
                    _logger.LogDebug("[ORDER_UPDATE] Received order update");
                });
            }

            if (_marketHubConnection != null)
            {
                _marketHubConnection.On<object>("MarketData", OnMarketDataReceived);
                _marketHubConnection.On<object>("ContractQuotes", OnMarketDataReceived);
            }
        }

        private Task PerformSystemReadinessChecksAsync()
        {
            _logger.LogInformation("🔍 Performing system readiness checks...");
            
            // Check connections
            var userHubReady = _userHubConnection?.State == HubConnectionState.Connected;
            var marketHubReady = _marketHubConnection?.State == HubConnectionState.Connected;
            
            _logger.LogInformation("📊 System Status - UserHub: {UserHub}, MarketHub: {MarketHub}, BarsSeen: {BarsSeen}",
                userHubReady, marketHubReady, _barsSeen);
                
            return Task.CompletedTask;
        }

        private Task MonitorSystemHealthAsync()
        {
            // Health monitoring implementation
            var healthScore = CalculateHealthScore();
            
            if (healthScore < 0.7)
            {
                _logger.LogWarning("⚠️ System health degraded: {HealthScore:F2}", healthScore);
            }
            
            return Task.CompletedTask;
        }

        private double CalculateHealthScore()
        {
            var score = 1.0;
            
            // Reduce score for disconnected hubs
            if (_userHubConnection?.State != HubConnectionState.Connected) score -= 0.3;
            if (_marketHubConnection?.State != HubConnectionState.Connected) score -= 0.3;
            
            // Reduce score for stale data
            if ((DateTime.UtcNow - _lastMarketDataUpdate).TotalMinutes > 5) score -= 0.2;
            
            // Reduce score for insufficient bars
            if (_barsSeen < 10) score -= 0.2;
            
            return Math.Max(0, score);
        }

        private void OnEmergencyStopTriggered(object? sender, EmergencyStopEventArgs e)
        {
            _logger.LogCritical("🚨 EMERGENCY STOP TRIGGERED - All trading halted. Reason: {Reason}", e.Reason);
            _isTradingEnabled = false;
        }

        private async Task CleanupAsync()
        {
            _logger.LogInformation("🧹 Cleaning up trading system resources...");
            
            _tradingEvaluationTimer?.Dispose();
            
            if (_userHubConnection != null)
            {
                await _userHubConnection.DisposeAsync();
            }
            
            if (_marketHubConnection != null)
            {
                await _marketHubConnection.DisposeAsync();
            }
            
            _orderConfirmation?.Dispose();
            
            _logger.LogInformation("✅ Trading system cleanup completed");
        }
    }

    /// <summary>
    /// Trading system status for monitoring
    /// </summary>
    public class TradingSystemStatus
    {
        public bool IsSystemReady { get; set; }
        public bool IsTradingEnabled { get; set; }
        public bool IsEmergencyStop { get; set; }
        public bool IsDryRunMode { get; set; }
        public double HealthScore { get; set; }
        public int ComponentCount { get; set; }
        public int CriticalComponents { get; set; }
        public DateTime LastUpdate { get; set; }
        public int BarsSeen { get; set; }
    }
}