using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using BotCore.Auth;
using System.Text.Json;
using Trading.Safety;

namespace BotCore.Services;

/// <summary>
/// Automatic TopstepX login service that handles authentication on startup
/// </summary>
public class AutoTopstepXLoginService : BackgroundService
{
    private readonly ILogger<AutoTopstepXLoginService> _logger;
    private readonly TopstepXCredentialManager _credentialManager;
    private readonly TopstepAuthAgent _authAgent;
    private readonly HttpClient _httpClient;

    public bool IsAuthenticated { get; private set; }
    public string? JwtToken { get; private set; }
    public string? AccountId { get; private set; }
    public TopstepXCredentials? CurrentCredentials { get; private set; }

    public AutoTopstepXLoginService(
        ILogger<AutoTopstepXLoginService> logger,
        TopstepXCredentialManager credentialManager,
        TopstepAuthAgent authAgent,
        HttpClient httpClient)
    {
        _logger = logger;
        _credentialManager = credentialManager;
        _authAgent = authAgent;
        _httpClient = httpClient;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try
        {
            await Task.Delay(2000, stoppingToken); // Wait for other services to start

            _logger.LogInformation("🚀 Starting automatic TopstepX login...");

            // Load credentials
            CurrentCredentials = _credentialManager.LoadCredentials();
            
            if (CurrentCredentials == null)
            {
                _logger.LogWarning("⚠️ No TopstepX credentials found - attempting auto-discovery...");
                await AttemptCredentialDiscovery(stoppingToken);
                return;
            }

            // Attempt login
            await AttemptLogin(stoppingToken);

            // If successful, start token refresh timer
            if (IsAuthenticated)
            {
                _ = Task.Run(async () => await TokenRefreshLoop(stoppingToken), stoppingToken);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error in AutoTopstepXLoginService");
        }
    }

    private async Task AttemptCredentialDiscovery(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🔍 Attempting to discover TopstepX credentials...");

        // Check common environment variable patterns
        var patterns = new[]
        {
            ("TOPSTEPX_USER", "TOPSTEPX_KEY"),
            ("TOPSTEP_USERNAME", "TOPSTEP_APIKEY"),
            ("TSX_USERNAME", "TSX_API_KEY"),
            ("TRADING_USERNAME", "TRADING_API_KEY")
        };

        foreach (var (userVar, keyVar) in patterns)
        {
            var username = Environment.GetEnvironmentVariable(userVar);
            var apiKey = Environment.GetEnvironmentVariable(keyVar);

            if (!string.IsNullOrEmpty(username) && !string.IsNullOrEmpty(apiKey))
            {
                _logger.LogInformation("✅ Found credentials in environment variables");
                
                CurrentCredentials = new TopstepXCredentials
                {
                    Username = username,
                    ApiKey = apiKey,
                    Source = $"Environment-{userVar}"
                };

                await AttemptLogin(cancellationToken);
                return;
            }
        }

        _logger.LogWarning("⚠️ No credentials discovered - bot will run in demo mode");
    }

    private async Task AttemptLogin(CancellationToken cancellationToken)
    {
        if (CurrentCredentials == null) return;

        try
        {
            _logger.LogInformation("🔐 Attempting login to TopstepX...");

            // Add timeout for login attempt
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(TimeSpan.FromSeconds(30)); // 30-second timeout

            // Get JWT token
            JwtToken = await _authAgent.GetJwtAsync(
                CurrentCredentials.Username, 
                CurrentCredentials.ApiKey, 
                timeoutCts.Token);

            if (!string.IsNullOrEmpty(JwtToken))
            {
                _logger.LogInformation("✅ Successfully obtained JWT token");

                // Update credentials with JWT
                CurrentCredentials.JwtToken = JwtToken;
                CurrentCredentials.LastUpdated = DateTime.UtcNow;

                // Set environment variables for other services
                _credentialManager.SetEnvironmentCredentials(
                    CurrentCredentials.Username, 
                    CurrentCredentials.ApiKey, 
                    JwtToken);

                // Get account information
                await GetAccountInfo(cancellationToken);

                // Save updated credentials
                await _credentialManager.SaveCredentialsAsync(CurrentCredentials);

                IsAuthenticated = true;
                _logger.LogInformation("🎉 TopstepX login successful! Bot ready for paper trading.");
            }
            else
            {
                _logger.LogError("❌ Failed to obtain JWT token");
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            _logger.LogWarning("⏰ TopstepX login timeout - network issues or slow response");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ TopstepX login failed");
            // Don't expose internal error details in logs that might be visible to users
        }
    }

    private async Task GetAccountInfo(CancellationToken cancellationToken)
    {
        try
        {
            _httpClient.DefaultRequestHeaders.Authorization = 
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", JwtToken);

            var response = await _httpClient.GetAsync("/api/Account", cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync(cancellationToken);
                using var doc = JsonDocument.Parse(content);
                
                if (doc.RootElement.TryGetProperty("data", out var dataElement) && 
                    dataElement.ValueKind == JsonValueKind.Array &&
                    dataElement.GetArrayLength() > 0)
                {
                    var firstAccount = dataElement[0];
                    if (firstAccount.TryGetProperty("id", out var idProp))
                    {
                        AccountId = idProp.GetString();
                        CurrentCredentials!.AccountId = AccountId;
                        Environment.SetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID", AccountId);
                        
                        _logger.LogInformation("✅ Retrieved account information");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "⚠️ Failed to get account info, but login still successful");
        }
    }

    private async Task TokenRefreshLoop(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested && IsAuthenticated)
        {
            try
            {
                // Refresh token every 30 minutes
                await Task.Delay(TimeSpan.FromMinutes(30), cancellationToken);

                _logger.LogDebug("🔄 Refreshing TopstepX token...");

                var newToken = await _authAgent.ValidateAsync(cancellationToken);
                if (!string.IsNullOrEmpty(newToken))
                {
                    JwtToken = newToken;
                    CurrentCredentials!.JwtToken = newToken;
                    Environment.SetEnvironmentVariable("TOPSTEPX_JWT", newToken);
                    
                    _logger.LogDebug("✅ Token refreshed successfully");
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ Token refresh failed, will retry");
            }
        }
    }

    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🛑 Stopping AutoTopstepXLoginService");
        await base.StopAsync(cancellationToken);
    }
}
