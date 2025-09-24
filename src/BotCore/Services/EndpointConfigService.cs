using System;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Production implementation of endpoint configuration
    /// Replaces hardcoded API endpoints and connection settings
    /// </summary>
    public class EndpointConfigService : IEndpointConfig
    {
        private readonly IConfiguration _config;
        private readonly ILogger<EndpointConfigService> _logger;

        public EndpointConfigService(IConfiguration config, ILogger<EndpointConfigService> logger)
        {
            _config = config;
            _logger = logger;
        }

        public Uri GetTopstepXApiBaseUrl() => 
            new Uri(_config.GetValue("Endpoints:TopstepXApiBaseUrl", "https://api.topstepx.com"));

        public Uri GetTopstepXWebSocketUrl() => 
            new Uri(_config.GetValue("Endpoints:TopstepXWebSocketUrl", "wss://api.topstepx.com/ws"));

        public string GetMLServiceEndpoint() => 
            _config.GetValue("Endpoints:MLServiceEndpoint", "http://localhost:8080");

        public string GetDataFeedEndpoint() => 
            _config.GetValue("Endpoints:DataFeedEndpoint", "https://datafeed.tradingbot.local");

        public string GetRiskServiceEndpoint() => 
            _config.GetValue("Endpoints:RiskServiceEndpoint", "http://localhost:9000");

        public string GetCloudStorageEndpoint() => 
            _config.GetValue("Endpoints:CloudStorageEndpoint", "https://storage.tradingbot.local");

        public int GetConnectionTimeoutSeconds() => 
            _config.GetValue("Endpoints:ConnectionTimeoutSeconds", 30);

        public int GetRequestTimeoutSeconds() => 
            _config.GetValue("Endpoints:RequestTimeoutSeconds", 60);

        public int GetMaxRetryAttempts() => 
            _config.GetValue("Endpoints:MaxRetryAttempts", 3);

        public bool UseSecureConnections() => 
            _config.GetValue("Endpoints:UseSecureConnections", true);
    }
}