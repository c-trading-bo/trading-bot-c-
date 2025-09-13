using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;
using System.Net.Http;
using OrchestratorAgent.Execution;

namespace UnifiedOrchestrator.Services
{
    /// <summary>
    /// Background service that triggers backtesting and learning when markets are closed
    /// This allows the ML/RL system to learn from historical data
    /// </summary>
    public class BacktestLearningService : BackgroundService
    {
        private readonly ILogger<BacktestLearningService> _logger;
        private readonly IServiceProvider _serviceProvider;
        private readonly HttpClient _httpClient;

        public BacktestLearningService(
            ILogger<BacktestLearningService> logger,
            IServiceProvider serviceProvider,
            HttpClient httpClient)
        {
            _logger = logger;
            _serviceProvider = serviceProvider;
            _httpClient = httpClient;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            _logger.LogInformation("🤖 [BACKTEST_LEARNING] Service starting...");

            // Wait for system to initialize
            await Task.Delay(10000, stoppingToken);

            var runLearning = Environment.GetEnvironmentVariable("RUN_LEARNING");
            var backtestMode = Environment.GetEnvironmentVariable("BACKTEST_MODE");
            
            if (runLearning != "1" && backtestMode != "1")
            {
                _logger.LogInformation("🤖 [BACKTEST_LEARNING] Not enabled (RUN_LEARNING={Learning}, BACKTEST_MODE={Backtest})", runLearning, backtestMode);
                return;
            }

            _logger.LogInformation("🚀 [BACKTEST_LEARNING] Starting historical data learning session...");

            try
            {
                await RunBacktestingSession(stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [BACKTEST_LEARNING] Backtesting session failed");
            }
        }

        private async Task RunBacktestingSession(CancellationToken cancellationToken)
        {
            _logger.LogInformation("📊 [BACKTEST_LEARNING] Initializing backtesting with historical data...");

            // Get JWT token function
            Func<Task<string>> getJwt = () =>
            {
                var jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                if (string.IsNullOrEmpty(jwt))
                {
                    throw new InvalidOperationException("TOPSTEPX_JWT environment variable not set");
                }
                return Task.FromResult(jwt);
            };

            // Use demo contract IDs for backtesting
            var esContractId = Environment.GetEnvironmentVariable("TOPSTEPX_EVAL_ES_ID") ?? "demo-es-contract";
            var nqContractId = Environment.GetEnvironmentVariable("TOPSTEPX_EVAL_NQ_ID") ?? "demo-nq-contract";

            // Define backtesting period (last 30 days)
            var endDate = DateTime.UtcNow.Date;
            var startDate = endDate.AddDays(-30);

            _logger.LogInformation("📈 [BACKTEST_LEARNING] Backtesting period: {Start} to {End}", startDate, endDate);

            try
            {
                // Run S2 strategy backtesting
                _logger.LogInformation("🔍 [BACKTEST_LEARNING] Running S2 strategy backtesting...");
                await TuningRunner.RunS2SummaryAsync(_httpClient, getJwt, esContractId, "ES", startDate, endDate, _logger, cancellationToken);

                // Wait a bit between runs
                await Task.Delay(2000, cancellationToken);

                // Run S3 strategy backtesting
                _logger.LogInformation("🔍 [BACKTEST_LEARNING] Running S3 strategy backtesting...");
                await TuningRunner.RunS3SummaryAsync(_httpClient, getJwt, nqContractId, "NQ", startDate, endDate, _logger, cancellationToken);

                _logger.LogInformation("✅ [BACKTEST_LEARNING] Backtesting session completed successfully");

                // Trigger adaptive learning
                _logger.LogInformation("🧠 [BACKTEST_LEARNING] Triggering adaptive learning from backtest results...");
                await TriggerAdaptiveLearning(cancellationToken);

            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [BACKTEST_LEARNING] Error during backtesting session");
            }
        }

        private async Task TriggerAdaptiveLearning(CancellationToken cancellationToken)
        {
            try
            {
                // The AdaptiveLearner will automatically read backtest results from state/backtest directory
                _logger.LogInformation("🧠 [BACKTEST_LEARNING] AdaptiveLearner will process backtest results automatically");
                _logger.LogInformation("📁 [BACKTEST_LEARNING] Backtest results saved to: state/backtest/");
                
                // Log current learning configuration
                var runLearning = Environment.GetEnvironmentVariable("RUN_LEARNING");
                var promoteTuner = Environment.GetEnvironmentVariable("PROMOTE_TUNER");
                var instantAllowLive = Environment.GetEnvironmentVariable("INSTANT_ALLOW_LIVE");
                
                _logger.LogInformation("⚙️ [BACKTEST_LEARNING] Learning config - RUN_LEARNING:{Learning} PROMOTE_TUNER:{Promote} INSTANT_ALLOW_LIVE:{Allow}", 
                    runLearning, promoteTuner, instantAllowLive);

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [BACKTEST_LEARNING] Error triggering adaptive learning");
            }
        }
    }
}