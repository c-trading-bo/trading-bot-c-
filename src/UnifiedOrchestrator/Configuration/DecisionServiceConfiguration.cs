namespace TradingBot.UnifiedOrchestrator.Configuration;

/// <summary>
/// Configuration options for DecisionServiceLauncher
/// </summary>
public class DecisionServiceLauncherOptions
{
    public string ExecutablePath { get; set; } = "decision-service";
    public string WorkingDirectory { get; set; } = "./decision-service";
    public int TimeoutSeconds { get; set; } = 300;
    public bool EnableLogging { get; set; } = true;
    public bool Enabled { get; set; } = false;
    public bool AutoRestart { get; set; } = true;
    public Dictionary<string, string> Environment { get; set; } = new();
}

/// <summary>
/// Configuration options for DecisionService
/// </summary>
public class DecisionServiceOptions
{
    public string ServiceUrl { get; set; } = "http://localhost:8080";
    public string BaseUrl { get; set; } = "http://localhost:8080";
    public string ApiKey { get; set; } = string.Empty;
    public int RequestTimeoutSeconds { get; set; } = 30;
    public int MaxRetries { get; set; } = 3;
    public bool EnableHealthChecks { get; set; } = true;
}

/// <summary>
/// Configuration options for DecisionServiceIntegration
/// </summary>
public class DecisionServiceIntegrationOptions
{
    public string IntegrationEndpoint { get; set; } = "http://localhost:8080/integration";
    public string WebhookUrl { get; set; } = string.Empty;
    public int SyncIntervalSeconds { get; set; } = 60;
    public bool EnableRealTimeSync { get; set; } = false;
    public bool Enabled { get; set; } = false;
    public int HealthCheckIntervalSeconds { get; set; } = 30;
    public bool LogDecisionLines { get; set; } = true;
    public bool EnableTradeManagement { get; set; } = false;
    public Dictionary<string, object> CustomSettings { get; set; } = new();
}

/// <summary>
/// Client for DecisionService
/// </summary>
public class DecisionServiceClient
{
    private readonly DecisionServiceOptions _options;
    private readonly HttpClient _httpClient;

    public DecisionServiceClient(DecisionServiceOptions options, HttpClient httpClient)
    {
        _options = options;
        _httpClient = httpClient;
        _httpClient.BaseAddress = new Uri(_options.ServiceUrl);
        _httpClient.Timeout = TimeSpan.FromSeconds(_options.RequestTimeoutSeconds);
    }

    public async Task<bool> IsHealthyAsync(CancellationToken cancellationToken = default)
    {
        if (!_options.EnableHealthChecks) return true;
        
        try
        {
            var response = await _httpClient.GetAsync("/health", cancellationToken);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task<string> GetDecisionAsync(string input, CancellationToken cancellationToken = default)
    {
        // Implementation would make actual decision service call
        await Task.Delay(100, cancellationToken); // Simulate call
        return "HOLD"; // Placeholder response
    }
}