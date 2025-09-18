using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TradingBot.Abstractions;
using Infrastructure.TopstepX;
using System.Text.Json;
using System.Net.Http.Json;
using TopstepX.Bot.Authentication;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// Automatic TopstepX login service that handles authentication on startup
/// </summary>
public class AutoTopstepXLoginService : BackgroundService
{
    #region Service Constants
    
    private const int STARTUP_DELAY_MS = 2000; // Wait for other services to start
    
    #endregion
    
    private readonly ILogger<AutoTopstepXLoginService> _logger;
    private readonly TopstepXCredentialManager _credentialManager;
    private readonly TopstepAuthAgent _authAgent;
    private readonly HttpClient _httpClient;
    private readonly ISignalRConnectionManager? _signalRConnectionManager;
    private readonly ILoginCompletionState _loginCompletionState;

    public bool IsAuthenticated { get; private set; }
    public string? JwtToken { get; private set; }
    public string? AccountId { get; private set; }
    public TopstepXCredentials? CurrentCredentials { get; private set; }

    public AutoTopstepXLoginService(
        ILogger<AutoTopstepXLoginService> logger,
        TopstepXCredentialManager credentialManager,
        TopstepAuthAgent authAgent,
        HttpClient httpClient,
        ISignalRConnectionManager? signalRConnectionManager = null,
        ILoginCompletionState? loginCompletionState = null)
    {
        _logger = logger;
        _credentialManager = credentialManager;
        _authAgent = authAgent;
        _httpClient = httpClient;
        _signalRConnectionManager = signalRConnectionManager;
        _loginCompletionState = loginCompletionState ?? new LoginCompletionState();
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try
        {
            await Task.Delay(STARTUP_DELAY_MS, stoppingToken); // Wait for other services to start

            _logger.LogInformation("🚀 Starting automatic TopstepX login...");

            // Enhanced credential discovery
            var credentialDiscovery = _credentialManager.DiscoverAllCredentialSources();
            
            if (!credentialDiscovery.HasAnyCredentials)
            {
                _logger.LogWarning("⚠️ No TopstepX credentials found in any source - attempting auto-discovery...");
                await AttemptCredentialDiscovery(stoppingToken);
                return;
            }

            // Use recommended credentials
            CurrentCredentials = credentialDiscovery.RecommendedCredentials;
            _logger.LogInformation("✅ Using credentials from: {Source} ({Total} sources found)", 
                credentialDiscovery.RecommendedSource, credentialDiscovery.TotalSourcesFound);

            if (credentialDiscovery.HasEnvironmentCredentials && credentialDiscovery.HasFileCredentials)
            {
                _logger.LogInformation("🔄 Both environment and file credentials found - preferring environment for automation");
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
                var credentialsSaved = await _credentialManager.SaveCredentialsAsync(CurrentCredentials);
                if (!credentialsSaved)
                {
                    _logger.LogWarning("⚠️ Login successful but credentials could not be saved to disk. " +
                                      "You may need to re-authenticate on next startup. " +
                                      "Check file permissions for the credentials directory.");
                    _credentialManager.LogCredentialSetupInstructions();
                }

                IsAuthenticated = true;
                _logger.LogInformation("🎉 TopstepX login successful! Bot ready for paper trading.");

                // ** CRITICAL: Signal that login is complete **
                _loginCompletionState.SetLoginCompleted();
            }
            else
            {
                _logger.LogError("❌ Failed to obtain JWT token");
            }
        }
        catch (OperationCanceledException ex) when (cancellationToken.IsCancellationRequested)
        {
            _logger.LogWarning(ex, "⏰ TopstepX login timeout - network issues or slow response");
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
            // Ensure BaseAddress is set - Use environment variable with fallback
            if (_httpClient.BaseAddress == null)
            {
                var apiBase = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
                _httpClient.BaseAddress = new Uri(apiBase);
            }
            
            _httpClient.DefaultRequestHeaders.Authorization = 
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", JwtToken);

            _logger.LogInformation("🔍 Searching for account information via TopstepX API...");
            
            // Enhanced: Search for ALL accounts (not just by account number)
            var searchRequest = new
            {
                // Remove accountNumber filter to get ALL accounts available to this user
                onlyActiveAccounts = true  // Only get active accounts per ProjectX API docs
            };

            var response = await _httpClient.PostAsJsonAsync("/api/Account/search", searchRequest, cancellationToken);
            
            _logger.LogInformation("📡 Account search response: {StatusCode} {ReasonPhrase}", response.StatusCode, response.ReasonPhrase);

            if (response.IsSuccessStatusCode)
            {
                var jsonResponse = await response.Content.ReadAsStringAsync(cancellationToken);
                
                // Log raw JSON for debugging
                _logger.LogTrace("Account search raw response: {JsonResponse}", jsonResponse);

                using var doc = JsonDocument.Parse(jsonResponse);
                
                if (!doc.RootElement.TryGetProperty("accounts", out var data))
                {
                    _logger.LogWarning("⚠️ Account search response does not contain 'accounts' property. Full response: {JsonResponse}", jsonResponse);
                    return;
                }

                if (data.ValueKind == JsonValueKind.Array && data.GetArrayLength() > 0)
                {
                    _logger.LogInformation("🔍 FOUND {Count} ACCOUNT(S) - Analyzing all available accounts...", data.GetArrayLength());

                    var accounts = new List<TopstepAccount>();
                    foreach (var accElement in data.EnumerateArray())
                    {
                        try
                        {
                            var account = new TopstepAccount
                            {
                                Id = accElement.TryGetProperty("id", out var id) && id.TryGetInt64(out var idVal) ? idVal : 0,
                                Name = accElement.TryGetProperty("name", out var name) ? name.GetString() : "N/A",
                                Type = accElement.TryGetProperty("type", out var type) ? type.GetString() : "N/A",
                                CanTrade = accElement.TryGetProperty("canTrade", out var canTrade) && canTrade.GetBoolean(),
                                Balance = accElement.TryGetProperty("balance", out var balance) && balance.TryGetDecimal(out var balVal) ? balVal : 0m
                            };
                            
                            if (account.Id == 0)
                            {
                                _logger.LogWarning("Skipping account with invalid or missing 'id'.");
                                continue;
                            }

                            accounts.Add(account);
                            _logger.LogInformation("📋 Account {Index}: ID={Id}, Name={Name}, Type={Type}, CanTrade={CanTrade}, Balance=${Balance}", 
                                accounts.Count, account.Id, account.Name, account.Type, account.CanTrade, account.Balance);
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning(ex, "Error parsing one of the accounts in the search response. JSON: {AccountJson}", accElement.ToString());
                        }
                    }

                    // Select the best account based on type (Trading Combine > Practice)
                    // Prioritize 50K Trading Combine account for live trading
                    var selectedAccount = accounts
                        .Where(a => a.CanTrade)
                        .OrderBy(a => a.Name?.Contains("50KTC") != true) // Prioritize 50K Trading Combine (50KTC)
                        .ThenBy(a => a.Name?.Contains("TRADING_COMBINE") != true)
                        .ThenBy(a => a.Name?.Contains("PRAC") == true) // Deprioritize practice accounts
                        .ThenByDescending(a => a.Balance)
                        .FirstOrDefault();

                    if (selectedAccount != null)
                    {
                        AccountId = selectedAccount.Id.ToString();
                        _logger.LogInformation("✅ SELECTED ACCOUNT (ID: {Id}, Type: {Type}) - Name: {Name}, CanTrade: {CanTrade}, Balance: ${Balance}", 
                            selectedAccount.Id, selectedAccount.Type, selectedAccount.Name, selectedAccount.CanTrade, selectedAccount.Balance);

                        // ** CRITICAL: Retry subscriptions now that we have the account ID **
                        if (_signalRConnectionManager != null)
                        {
                            _logger.LogInformation("🔄 Retrying SignalR subscriptions with account ID: {AccountId}", AccountId);
                            var subscribed = await _signalRConnectionManager.RetrySubscriptionsWithAccountId(AccountId);
                            if (!subscribed)
                            {
                                _logger.LogWarning("⚠️ SignalR subscriptions failed for account {AccountId}", AccountId);
                            }
                        }
                    }
                    else
                    {
                        _logger.LogWarning("⚠️ No suitable trading account with CanTrade=true found for this user.");
                    }
                }
                else
                {
                    _logger.LogWarning("⚠️ No trading accounts found for this user, or 'data' is not an array. Response: {JsonResponse}", jsonResponse);
                }
            }
            else
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken);
                _logger.LogError("❌ Failed to get account info. Status: {StatusCode}, Response: {ErrorContent}", 
                    response.StatusCode, errorContent);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error getting account info");
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

    // AsyncFixer false positive: This method overrides base.StopAsync and legitimately needs await
    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🛑 Stopping AutoTopstepXLoginService");
        await base.StopAsync(cancellationToken);
    }
}
