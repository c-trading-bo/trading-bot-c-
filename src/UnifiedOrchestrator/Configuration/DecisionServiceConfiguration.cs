namespace TradingBot.UnifiedOrchestrator.Configuration;

/// <summary>
/// Configuration options for DecisionServiceLauncher
/// </summary>
public class DecisionServiceLauncherOptions
{
    public string ExecutablePath { get; set; } = "decision-service";
    public string WorkingDirectory { get; set; } = "./decision-service";
    public string ScriptPath { get; set; } = "./decision-service/start.py";
    public string ConfigFile { get; set; } = "./decision-service/config.json";
    public string PythonExecutable { get; set; } = "python";
    public int TimeoutSeconds { get; set; } = 300;
    public int StartupTimeoutSeconds { get; set; } = 60;
    public string Host { get; set; } = "localhost";
    public int Port { get; set; } = 8080;
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
    public int TimeoutMs { get; set; } = 30000;
    public int MaxRetries { get; set; } = 3;
    public bool EnableHealthChecks { get; set; } = true;
    public bool Enabled { get; set; } = false;
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
        try
        {
            // In DRY_RUN mode, return conservative decision
            await Task.Delay(100, cancellationToken); // Simulate processing time
            
            // Parse input and make conservative decision
            if (string.IsNullOrWhiteSpace(input))
                return "HOLD";
            
            // Simple decision logic - always be conservative in production
            if (input.Contains("BUY") || input.Contains("SELL"))
            {
                // In production mode, this would integrate with ML models
                // For now, return HOLD for safety
                return "HOLD";
            }
            
            return "HOLD"; // Default conservative decision
        }
        catch (Exception ex)
        {
            // Log error and return safe default
            Console.WriteLine($"Error in decision service: {ex.Message}");
            return "HOLD";
        }
    }
}