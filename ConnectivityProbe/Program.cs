using Microsoft.Extensions.Logging;

var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
var logger = loggerFactory.CreateLogger<Program>();

try
{
    // Check for JWT token and basic connectivity
    var jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
    
    if (string.IsNullOrWhiteSpace(jwt))
    {
        logger.LogWarning("No TOPSTEPX_JWT found in environment variables");
        Environment.Exit(2); // Missing credentials
    }
    
    // Try to connect to TopstepX API endpoints
    using var httpClient = new HttpClient();
    httpClient.BaseAddress = new Uri("https://api.topstepx.com");
    httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {jwt}");
    httpClient.Timeout = TimeSpan.FromSeconds(10);
    
    try
    {
        var response = await httpClient.GetAsync("/api/v1/health");
        if (response.IsSuccessStatusCode)
        {
            logger.LogInformation("Connectivity probe passed");
            Environment.Exit(0); // Success
        }
        else
        {
            logger.LogWarning("API returned status: {StatusCode}", response.StatusCode);
            Environment.Exit(1); // Network/transport error
        }
    }
    catch (HttpRequestException ex)
    {
        logger.LogError(ex, "Network connectivity error");
        Environment.Exit(1); // Network/transport error
    }
    catch (TaskCanceledException ex)
    {
        logger.LogError(ex, "Request timeout");
        Environment.Exit(1); // Network/transport error
    }
}
catch (Exception ex)
{
    logger.LogError(ex, "Unexpected error during connectivity probe");
    Environment.Exit(1);
}
