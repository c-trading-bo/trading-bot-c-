using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of execution policy configuration
    /// Replaces hardcoded order type defaults and execution policies
    /// </summary>
    public class ExecutionPolicyConfigService : IExecutionPolicyConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<ExecutionPolicyConfigService> _logger;

        public ExecutionPolicyConfigService(IConfiguration config, ILogger<ExecutionPolicyConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public string GetDefaultEntryOrderType() => 
            _config.GetValue("ExecutionPolicy:DefaultEntryOrderType", "LIMIT");

        public string GetDefaultExitOrderType() => 
            _config.GetValue("ExecutionPolicy:DefaultExitOrderType", "MARKET");

        public bool UseAggressiveFillsDuringVolatility() => 
            _config.GetValue("ExecutionPolicy:UseAggressiveFillsDuringVolatility", true);

        public int GetLimitOrderTimeoutSeconds() => 
            _config.GetValue("ExecutionPolicy:LimitOrderTimeoutSeconds", 30);

        public bool EnableSmartOrderRouting() => 
            _config.GetValue("ExecutionPolicy:EnableSmartOrderRouting", false);

        public int GetMaxIcebergSize() => 
            _config.GetValue("ExecutionPolicy:MaxIcebergSize", 10);

        public int GetMinOrderSize() => 
            _config.GetValue("ExecutionPolicy:MinOrderSize", 1);
    }
}