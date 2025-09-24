using System;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of execution guards configuration
    /// Replaces hardcoded execution limits with configurable values
    /// </summary>
    public class ExecutionGuardsConfigService : IExecutionGuardsConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<ExecutionGuardsConfigService> _logger;

        public ExecutionGuardsConfigService(IConfiguration config, ILogger<ExecutionGuardsConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public double GetMaxSpreadTicks() => 
            _config.GetValue("ExecutionGuards:MaxSpreadTicks", 3.0);

        public int GetMaxLatencyMs() => 
            _config.GetValue("ExecutionGuards:MaxLatencyMs", 100);

        public long GetMinVolumeThreshold() => 
            _config.GetValue("ExecutionGuards:MinVolumeThreshold", 1000L);

        public double GetMaxImbalanceRatio() => 
            _config.GetValue("ExecutionGuards:MaxImbalanceRatio", 0.8);

        public double GetMaxLimitOffsetTicks() => 
            _config.GetValue("ExecutionGuards:MaxLimitOffsetTicks", 5.0);

        public int GetCircuitBreakerThreshold() => 
            _config.GetValue("ExecutionGuards:CircuitBreakerThreshold", 10);
    }
}