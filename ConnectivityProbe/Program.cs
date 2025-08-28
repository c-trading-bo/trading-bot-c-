using Microsoft.Extensions.Logging;

namespace ConnectivityProbe;

class Program
{
    static async Task<int> Main(string[] args)
    {
        using var loggerFactory = LoggerFactory.Create(builder =>
        {
            builder
                .AddConsole()
                .SetMinimumLevel(LogLevel.Information);
        });

        var logger = loggerFactory.CreateLogger<Program>();
        
        logger.LogInformation("ConnectivityProbe: Starting connectivity check...");

        try
        {
            // Check for basic JWT/credentials
            var jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
            var apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");

            if (string.IsNullOrWhiteSpace(jwt) && 
                (string.IsNullOrWhiteSpace(username) || string.IsNullOrWhiteSpace(apiKey)))
            {
                logger.LogWarning("ConnectivityProbe: Missing JWT/login credentials");
                return 2; // Missing credentials, but not fatal
            }

            // Simple connectivity check to TopstepX API
            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromSeconds(10);
            
            try
            {
                var response = await client.GetAsync("https://api.topstepx.com/healthz");
                if (response.IsSuccessStatusCode)
                {
                    logger.LogInformation("ConnectivityProbe: Connectivity check passed");
                    return 0; // Success
                }
                else
                {
                    logger.LogWarning("ConnectivityProbe: API returned status {StatusCode}", response.StatusCode);
                    return 1; // Transport/network error
                }
            }
            catch (HttpRequestException ex)
            {
                logger.LogError(ex, "ConnectivityProbe: Network error during connectivity check");
                return 1; // Transport/network error
            }
            catch (TaskCanceledException ex)
            {
                logger.LogError(ex, "ConnectivityProbe: Timeout during connectivity check");
                return 1; // Transport/network error
            }
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "ConnectivityProbe: Unexpected error");
            return 1; // Transport/network error
        }
    }
}