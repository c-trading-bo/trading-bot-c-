using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.ML;
using TradingBot.ML;
using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace TradingBot.Core
{
    /// <summary>
    /// 🚀 MASTER ORCHESTRATOR - UNIFIED BRAIN FOR ALL TRADING OPERATIONS
    /// Replaces all fragmented orchestrators with one unified coordinator
    /// </summary>
    public class MasterOrchestrator : BackgroundService
    {
        private readonly ILogger<MasterOrchestrator> _logger;
        private readonly IServiceProvider _serviceProvider;
        private readonly SharedSystemState _sharedState;
        private readonly SemaphoreSlim _coordinationLock = new(1, 1);
        
        // Core Components
        private DataComponent _dataComponent;
        private IntelligenceComponent _intelligenceComponent;
        private TradingComponent _tradingComponent;
        
        public MasterOrchestrator(
            ILogger<MasterOrchestrator> logger,
            IServiceProvider serviceProvider,
            SharedSystemState sharedState)
        {
            _logger = logger;
            _serviceProvider = serviceProvider;
            _sharedState = sharedState;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("🚀 MASTER ORCHESTRATOR STARTING - ALL SYSTEMS INITIALIZING");
            
            try
            {
                // Initialize all components
                await InitializeComponents(stoppingToken);
                
                // Reset UCB daily stats
                var ucb = _serviceProvider.GetRequiredService<UCBManager>();
                await ucb.ResetDailyAsync();
                _logger.LogInformation("✅ UCB daily stats reset");
                
                // Health check UCB API
                if (!await ucb.IsHealthyAsync(stoppingToken))
                {
                    _logger.LogCritical("UCB API not reachable at startup");
                    throw new InvalidOperationException("UCB API not reachable");
                }
                
                _logger.LogInformation("✅ All components initialized");
                
                // Main orchestration loop
                while (!stoppingToken.IsCancellationRequested)
                {
                    await _coordinationLock.WaitAsync(stoppingToken);
                    try
                    {
                        await RunOrchestrationCycle(stoppingToken);
                    }
                    finally
                    {
                        _coordinationLock.Release();
                    }
                    
                    // Log system state every cycle
                    var state = _sharedState.GetCurrentState();
                    _logger.LogInformation("System State: PnL={DailyPnL:F2} | Drawdown={Drawdown:F2} | Positions={OpenPositions} | Mode={TradingMode}",
                        state.DailyPnL, state.Drawdown, state.OpenPositions, state.TradingMode);
                    
                    await Task.Delay(TimeSpan.FromSeconds(1), stoppingToken);
                }
            }
            catch (Exception ex)
            {
                _logger.LogCritical(ex, "CRITICAL ERROR in Master Orchestrator");
                throw;
            }
        }

        private async Task InitializeComponents(CancellationToken stoppingToken)
        {
            _dataComponent = new DataComponent(_serviceProvider, _sharedState);
            _intelligenceComponent = new IntelligenceComponent(_serviceProvider, _sharedState);
            _tradingComponent = new TradingComponent(_serviceProvider, _sharedState);
            
            await _dataComponent.InitializeAsync();
            await _intelligenceComponent.InitializeAsync();
            await _tradingComponent.InitializeAsync();
        }

        private async Task RunOrchestrationCycle(CancellationToken stoppingToken)
        {
            // 1. Gather all market data
            var marketData = await _dataComponent.GatherAllDataAsync(stoppingToken);
            _sharedState.UpdateMarketData(marketData);
            
            // 2. Run intelligence analysis
            var analysis = await _intelligenceComponent.AnalyzeAsync(marketData, stoppingToken);
            _sharedState.UpdateAnalysis(analysis);
            
            // 3. Check and enforce risk limits
            await CheckAndEnforceRiskLimits();
            
            // 4. Execute trading logic
            await _tradingComponent.ExecuteAsync(analysis, stoppingToken);
        }

        private async Task CheckAndEnforceRiskLimits()
        {
            var ucb = _serviceProvider.GetRequiredService<UCBManager>();
            var limits = await ucb.CheckLimits();
            
            if (!limits.CanTrade)
            {
                _logger.LogError("Trading blocked: {Reason} | PnL={DailyPnL:F2} | DD={Drawdown:F2}",
                    limits.Reason, limits.DailyPnL, limits.CurrentDrawdown);
                _sharedState.SetTradingMode(TradingMode.Stopped);
                return;
            }
            
            if (limits.DailyPnL <= -900m)
            {
                _logger.LogWarning("Approaching daily loss limit: {DailyPnL:F2}", limits.DailyPnL);
                _sharedState.SetTradingMode(TradingMode.Conservative);
            }
            else if (_sharedState.TradingMode != TradingMode.Stopped && limits.DailyPnL > -900m)
            {
                _sharedState.SetTradingMode(TradingMode.Normal);
            }
            
            // USE THE NEW METHOD - NO LOCK NEEDED (it's internal)
            _sharedState.SetRiskSnapshot(limits.DailyPnL, limits.CurrentDrawdown);
        }

        public override void Dispose()
        {
            _coordinationLock?.Dispose();
            base.Dispose();
        }
    }

    // Component implementations
    public class DataComponent
    {
        private readonly MarketDataAggregator _aggregator;
        private readonly DataFeedRedundancyService _redundancy;
        private readonly IndicatorCalculator _indicators;
        private readonly SharedSystemState _sharedState;

        public DataComponent(IServiceProvider services, SharedSystemState sharedState)
        {
            _aggregator = services.GetRequiredService<MarketDataAggregator>();
            _redundancy = services.GetRequiredService<DataFeedRedundancyService>();
            _indicators = services.GetRequiredService<IndicatorCalculator>();
            _sharedState = sharedState;
        }

        public async Task InitializeAsync()
        {
            await _redundancy.ConnectAllFeedsAsync();
        }

        // RETURNS Core.MarketData (not Data.MarketData)
        public async Task<TradingBot.Core.MarketData> GatherAllDataAsync(CancellationToken ct)
        {
            // Get raw data from redundancy service
            var raw = await _redundancy.GetBestDataAsync();
            
            // Calculate all indicators
            var indicators = await _indicators.CalculateAll();
            
            // Build Core.MarketData
            return new TradingBot.Core.MarketData
            {
                ESPrice = raw.ESPrice,
                NQPrice = raw.NQPrice,
                ESVolume = raw.ESVolume,
                NQVolume = raw.NQVolume,
                Indicators = indicators,
                Timestamp = DateTime.UtcNow,
                Internals = new TradingBot.Core.MarketInternals
                {
                    VIX = 15m,  // Get from data provider
                    TICK = 0,
                    ADD = 0,
                    VOLD = 0
                },
                Correlation = 0.85m,  // Calculate or get from correlations service
                PrimaryInstrument = "ES"
            };
        }
    }

    public class IntelligenceComponent
    {
        private readonly UCBManager _ucbManager;
        private readonly SharedSystemState _sharedState;

        public IntelligenceComponent(IServiceProvider services, SharedSystemState sharedState)
        {
            _ucbManager = services.GetRequiredService<UCBManager>();
            _sharedState = sharedState;
        }

        public async Task InitializeAsync()
        {
            // Initialize AI components
        }

        public async Task<MarketAnalysis> AnalyzeAsync(MarketData data, CancellationToken ct)
        {
            // Get UCB recommendations
            var recommendation = await _ucbManager.GetRecommendationAsync(data.ToMLMarketData(), ct);
            
            return new MarketAnalysis
            {
                Timestamp = DateTime.UtcNow,
                UCBAction = recommendation.Action,
                UCBConfidence = recommendation.Confidence,
                Signal = recommendation.Action == "BUY" ? SignalType.Long : 
                        recommendation.Action == "SELL" ? SignalType.Short : SignalType.Hold
            };
        }
    }

    public class TradingComponent
    {
        private readonly SharedSystemState _sharedState;

        public TradingComponent(IServiceProvider services, SharedSystemState sharedState)
        {
            _sharedState = sharedState;
        }

        public async Task InitializeAsync()
        {
            // Initialize trading components
        }

        public async Task ExecuteAsync(MarketAnalysis analysis, CancellationToken ct)
        {
            // Trading execution logic based on analysis
            // This preserves all your existing trading strategies
        }
    }

    // Supporting types
    public enum TradingMode
    {
        Normal,
        Conservative, 
        Stopped
    }

    public enum SignalType
    {
        Hold,
        Long,
        Short
    }

    public class SystemState
    {
        public decimal DailyPnL { get; set; }
        public decimal Drawdown { get; set; }
        public int OpenPositions { get; set; }
        public TradingMode TradingMode { get; set; }
    }

    public class MarketAnalysis
    {
        public DateTime Timestamp { get; set; }
        public string UCBAction { get; set; }
        public double UCBConfidence { get; set; }
        public SignalType Signal { get; set; }
    }

    public class RiskLimits
    {
        public decimal DailyLoss { get; set; }
        public decimal MaxDrawdown { get; set; }
    }

    // Placeholder services - these would be implemented based on your existing components
    public class MarketDataAggregator
    {
        public async Task<RawMarketData> GetBestDataAsync() => new RawMarketData();
    }

    public class IndicatorCalculator
    {
        public async Task<Dictionary<string, double>> CalculateAll() => new Dictionary<string, double>();
    }

    public class RawMarketData
    {
        public decimal ESPrice { get; set; }
        public decimal NQPrice { get; set; }
        public long ESVolume { get; set; }
        public long NQVolume { get; set; }
    }
}
