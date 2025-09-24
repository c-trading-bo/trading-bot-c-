namespace TradingBot.Abstractions
{
    /// <summary>
    /// Configuration interface for API endpoints and connectivity settings
    /// Replaces hardcoded endpoint URLs and connection parameters
    /// </summary>
    public interface IEndpointConfig
    {
        /// <summary>
        /// TopstepX API base URL
        /// </summary>
        string GetTopstepXApiBaseUrl();

        /// <summary>
        /// TopstepX WebSocket URL for real-time data
        /// </summary>
        string GetTopstepXWebSocketUrl();

        /// <summary>
        /// ML service endpoint URL
        /// </summary>
        string GetMLServiceEndpoint();

        /// <summary>
        /// Data feed endpoint URL
        /// </summary>
        string GetDataFeedEndpoint();

        /// <summary>
        /// Risk management service endpoint
        /// </summary>
        string GetRiskServiceEndpoint();

        /// <summary>
        /// Cloud storage endpoint for models and logs
        /// </summary>
        string GetCloudStorageEndpoint();

        /// <summary>
        /// Connection timeout in seconds
        /// </summary>
        int GetConnectionTimeoutSeconds();

        /// <summary>
        /// Request timeout in seconds
        /// </summary>
        int GetRequestTimeoutSeconds();

        /// <summary>
        /// Maximum retry attempts for failed requests
        /// </summary>
        int GetMaxRetryAttempts();

        /// <summary>
        /// Whether to use SSL/TLS for connections
        /// </summary>
        bool UseSecureConnections();
    }
}