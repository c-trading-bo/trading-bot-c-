using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.ML;
using TradingBot.UnifiedOrchestrator.Services;
using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

namespace TradingBot.UnifiedOrchestrator.Services
{
    /// <summary>
    /// 🚀 MASTER ORCHESTRATOR - UNIFIED BRAIN FOR ALL TRADING OPERATIONS
    /// Replaces all fragmented orchestrators with one unified coordinator
    /// </summary>
    public class MasterOrchestrator : BackgroundService
    {
        private readonly ILogger<MasterOrchestrator> _logger;
        private readonly IServiceProvider _serviceProvider;
        private readonly SemaphoreSlim _coordinationLock = new(1, 1);

        // Components
        private DataComponent? _dataComponent;
        private IntelligenceComponent? _intelligenceComponent;
        private TradingComponent? _tradingComponent;
        private SharedSystemState? _sharedState;

        public MasterOrchestrator(
            ILogger<MasterOrchestrator> logger,
            IServiceProvider serviceProvider)
        {
            _logger = logger;
            _serviceProvider = serviceProvider;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("🚀 MASTER ORCHESTRATOR STARTING - ALL SYSTEMS INITIALIZING");

            try
            {
                // Initialize all components
                await InitializeAllComponentsAsync(stoppingToken);

                // Reset UCB daily stats at startup
                await ResetUCBDailyStatsAsync(stoppingToken);

                _logger.LogInformation("✅ All components initialized");

                // Main coordination loop
                await RunCoordinationLoopAsync(stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogCritical(ex, "Master Orchestrator failed");
                throw;
            }
        }

        private async Task InitializeAllComponentsAsync(CancellationToken stoppingToken)
        {
            await _coordinationLock.WaitAsync(stoppingToken);
            try
            {
                _sharedState = new SharedSystemState();
                _dataComponent = new DataComponent(_serviceProvider, _sharedState);
                _intelligenceComponent = new IntelligenceComponent(_serviceProvider, _sharedState);
                _tradingComponent = new TradingComponent(_serviceProvider, _sharedState);

                // Initialize each component
                await _dataComponent.InitializeAsync();
                await _intelligenceComponent.InitializeAsync();
                await _tradingComponent.InitializeAsync();
            }
            finally
            {
                _coordinationLock.Release();
            }
        }

        private async Task ResetUCBDailyStatsAsync(CancellationToken stoppingToken)
        {
            try
            {
                var ucb = _serviceProvider.GetRequiredService<UCBManager>();
                // Call the health check to ensure UCB is running
                await ucb.CheckLimits();
                _logger.LogInformation("✅ UCB service verified and ready");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "UCB verification failed - continuing without UCB");
            }
        }

        private async Task RunCoordinationLoopAsync(CancellationToken stoppingToken)
        {
            var lastRiskCheck = DateTime.UtcNow.AddMinutes(-5); // Force immediate check
            var lastStateLog = DateTime.UtcNow.AddMinutes(-1);

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    var now = DateTime.UtcNow;

                    // 1. Gather market data
                    var marketData = await _dataComponent!.GatherAllDataAsync(stoppingToken);
                    _sharedState!.UpdateMarketData(marketData);

                    // 2. Run intelligence analysis
                    var analysis = await _intelligenceComponent!.RunAnalysisAsync(marketData, stoppingToken);
                    _sharedState.UpdateAnalysis(analysis);

                    // 3. Execute trading decisions
                    await _tradingComponent!.ExecuteTradingDecisionsAsync(analysis, stoppingToken);

                    // 4. Risk management (every 30 seconds)
                    if (now - lastRiskCheck > TimeSpan.FromSeconds(30))
                    {
                        await CheckAndEnforceRiskLimits();
                        lastRiskCheck = now;
                    }

                    // 5. Log system state (every 60 seconds)
                    if (now - lastStateLog > TimeSpan.FromMinutes(1))
                    {
                        LogSystemState();
                        lastStateLog = now;
                    }

                    // Coordination interval
                    await Task.Delay(1000, stoppingToken);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Coordination loop error");
                    await Task.Delay(5000, stoppingToken);
                }
            }
        }

        private async Task CheckAndEnforceRiskLimits()
        {
            var ucb = _serviceProvider.GetRequiredService<UCBManager>();
            var limits = await ucb.CheckLimits();

            if (!limits.CanTrade)
            {
                _logger.LogError("Trading blocked: {Reason} | PnL={DailyPnL:F2} | DD={Drawdown:F2}",
                    limits.Reason, limits.DailyPnL, limits.CurrentDrawdown);
                _sharedState!.SetTradingMode(TradingMode.Stopped);
                return;
            }

            if (limits.DailyPnL <= -900m)
            {
                _logger.LogWarning("Approaching daily loss limit: {DailyPnL:F2}", limits.DailyPnL);
                _sharedState!.SetTradingMode(TradingMode.Conservative);
            }
            else if (_sharedState!.TradingMode != TradingMode.Stopped && limits.DailyPnL > -900m)
            {
                _sharedState.SetTradingMode(TradingMode.Normal);
            }

            // USE THE NEW METHOD - NO LOCK NEEDED (it's internal)
            _sharedState.SetRiskSnapshot(limits.DailyPnL, limits.CurrentDrawdown);
        }

        private void LogSystemState()
        {
            var state = _sharedState!.GetCurrentState();
            _logger.LogInformation("System State: PnL={DailyPnL:F2} | Drawdown={Drawdown:F2} | Positions={OpenPositions} | Mode={TradingMode}",
                state.DailyPnL, state.Drawdown, state.OpenPositions, state.TradingMode);
        }

        public override void Dispose()
        {
            _coordinationLock?.Dispose();
            base.Dispose();
        }
    }

    // Component classes
    public class DataComponent
    {
        private readonly IServiceProvider _serviceProvider;
        private readonly SharedSystemState _sharedState;

        public DataComponent(IServiceProvider services, SharedSystemState sharedState)
        {
            _serviceProvider = services;
            _sharedState = sharedState;
        }

        public Task InitializeAsync() => Task.CompletedTask;

        public Task<TradingBot.UnifiedOrchestrator.Models.MarketData> GatherAllDataAsync(CancellationToken ct)
        {
            // Simple market data for now
            return Task.FromResult(new TradingBot.UnifiedOrchestrator.Models.MarketData
            {
                ESPrice = 5530m,
                NQPrice = 19250m,
                ESVolume = 100000,
                NQVolume = 75000,
                Timestamp = DateTime.UtcNow,
                Correlation = 0.85m,
                PrimaryInstrument = "ES"
            });
        }
    }

    public class IntelligenceComponent
    {
        private readonly IServiceProvider _serviceProvider;
        private readonly SharedSystemState _sharedState;

        public IntelligenceComponent(IServiceProvider services, SharedSystemState sharedState)
        {
            _serviceProvider = services;
            _sharedState = sharedState;
        }

        public Task InitializeAsync() => Task.CompletedTask;

        public Task<MarketAnalysis> RunAnalysisAsync(TradingBot.UnifiedOrchestrator.Models.MarketData marketData, CancellationToken ct)
        {
            // Simple analysis for now
            return Task.FromResult(new MarketAnalysis
            {
                Timestamp = DateTime.UtcNow,
                Recommendation = "HOLD",
                Confidence = 0.7,
                ExpectedDirection = "NEUTRAL"
            });
        }
    }

    public class TradingComponent
    {
        private readonly IServiceProvider _serviceProvider;
        private readonly SharedSystemState _sharedState;

        public TradingComponent(IServiceProvider services, SharedSystemState sharedState)
        {
            _serviceProvider = services;
            _sharedState = sharedState;
        }

        public Task InitializeAsync() => Task.CompletedTask;

        public Task ExecuteTradingDecisionsAsync(MarketAnalysis analysis, CancellationToken ct)
        {
            // Trading logic will be implemented here
            return Task.CompletedTask;
        }
    }

    // Supporting classes
    public class MarketAnalysis
    {
        public DateTime Timestamp { get; set; }
        public string Recommendation { get; set; } = "HOLD";
        public double Confidence { get; set; }
        public string ExpectedDirection { get; set; } = "NEUTRAL";
    }

    public enum TradingMode
    {
        Normal,
        Conservative,
        Stopped
    }

    public class SystemState
    {
        public decimal DailyPnL { get; set; }
        public decimal Drawdown { get; set; }
        public int OpenPositions { get; set; }
        public TradingMode TradingMode { get; set; }
    }

    public class SharedSystemState
    {
        private readonly object _lock = new();
        
        public TradingBot.UnifiedOrchestrator.Models.MarketData? CurrentMarketData { get; private set; }
        public MarketAnalysis? CurrentAnalysis { get; private set; }
        public decimal DailyPnL { get; private set; }
        public decimal Drawdown { get; private set; }
        public TradingMode TradingMode { get; private set; } = TradingMode.Normal;

        public void UpdateMarketData(TradingBot.UnifiedOrchestrator.Models.MarketData data)
        {
            lock (_lock) CurrentMarketData = data;
        }

        public void UpdateAnalysis(MarketAnalysis analysis)
        {
            lock (_lock) CurrentAnalysis = analysis;
        }

        public void SetTradingMode(TradingMode mode)
        {
            lock (_lock) TradingMode = mode;
        }

        public void SetRiskSnapshot(decimal dailyPnL, decimal drawdown)
        {
            lock (_lock)
            {
                DailyPnL = dailyPnL;
                Drawdown = drawdown;
            }
        }

        public SystemState GetCurrentState()
        {
            lock (_lock)
            {
                return new SystemState
                {
                    DailyPnL = DailyPnL,
                    Drawdown = Drawdown,
                    OpenPositions = 0,
                    TradingMode = TradingMode
                };
            }
        }
    }
}
