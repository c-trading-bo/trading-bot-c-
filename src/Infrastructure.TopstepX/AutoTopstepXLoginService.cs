using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TradingBot.Abstractions;
using Infrastructure.TopstepX;
using System.Text.Json;
using System.Net.Http.Json;
// // using Trading.Safety;

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
    private readonly ISignalRConnectionManager? _signalRConnectionManager;

    public bool IsAuthenticated { get; private set; }
    public string? JwtToken { get; private set; }
    public string? AccountId { get; private set; }
    public TopstepXCredentials? CurrentCredentials { get; private set; }

    public AutoTopstepXLoginService(
        ILogger<AutoTopstepXLoginService> logger,
        TopstepXCredentialManager credentialManager,
        TopstepAuthAgent authAgent,
        HttpClient httpClient,
        ISignalRConnectionManager? signalRConnectionManager = null)
    {
        _logger = logger;
        _credentialManager = credentialManager;
        _authAgent = authAgent;
        _httpClient = httpClient;
        _signalRConnectionManager = signalRConnectionManager;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        try
        {
            await Task.Delay(2000, stoppingToken); // Wait for other services to start

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
            // Ensure BaseAddress is set - UPDATED: Use working TopstepX API
            if (_httpClient.BaseAddress == null)
            {
                _httpClient.BaseAddress = new Uri("https://api.topstepx.com");
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
            
            var searchContent = JsonContent.Create(searchRequest);
            var searchResponse = await _httpClient.PostAsync("/api/Account/search", searchContent, cancellationToken);
            
            _logger.LogInformation("📡 Account search response: {StatusCode} {ReasonPhrase}", 
                searchResponse.StatusCode, searchResponse.ReasonPhrase);
            
            if (searchResponse.IsSuccessStatusCode)
            {
                var searchResult = await searchResponse.Content.ReadAsStringAsync(cancellationToken);
                _logger.LogDebug("📄 Account search response content: {Content}", searchResult);
                
                using var searchDoc = JsonDocument.Parse(searchResult);
                
                if (searchDoc.RootElement.TryGetProperty("accounts", out var accountsElement) && 
                    accountsElement.ValueKind == JsonValueKind.Array &&
                    accountsElement.GetArrayLength() > 0)
                {
                    _logger.LogInformation("🔍 FOUND {Count} ACCOUNT(S) - Analyzing all available accounts...", accountsElement.GetArrayLength());
                    
                    // Log ALL available accounts to help identify the Trading Combine account
                    var accountIndex = 0;
                    foreach (var account in accountsElement.EnumerateArray())
                    {
                        accountIndex++;
                        var accountId = account.TryGetProperty("id", out var idProp) ? idProp.GetInt64().ToString() : "Unknown";
                        var accountName = account.TryGetProperty("name", out var nameProp) ? nameProp.GetString() : "Unknown";
                        var canTrade = account.TryGetProperty("canTrade", out var canTradeProp) ? canTradeProp.GetBoolean() : false;
                        var balance = account.TryGetProperty("balance", out var balProp) ? balProp.GetDecimal() : 0m;
                        var accountType = accountName?.Contains("PRAC") == true ? "PRACTICE" : 
                                         accountName?.Contains("TST") == true ? "TEST" : 
                                         balance == 50000m ? "TRADING_COMBINE_50K" : "LIVE";
                        
                        _logger.LogInformation("📋 Account {Index}: ID={AccountId}, Name={AccountName}, Type={AccountType}, CanTrade={CanTrade}, Balance=${Balance:F2}", 
                            accountIndex, accountId, accountName, accountType, canTrade, balance);
                    }
                    
                    // ENHANCED: Look for Trading Combine account (50K balance) first, then any tradable account
                    JsonElement? tradingCombineAccount = null;
                    JsonElement? tradableAccount = null;
                    
                    foreach (var account in accountsElement.EnumerateArray())
                    {
                        var canTrade2 = account.TryGetProperty("canTrade", out var canTradeProp2) && canTradeProp2.GetBoolean();
                        var balance2 = account.TryGetProperty("balance", out var balProp2) ? balProp2.GetDecimal() : 0m;
                        var accountName2 = account.TryGetProperty("name", out var nameProp2) ? nameProp2.GetString() : "";
                        
                        // Prioritize Trading Combine account (50K balance, non-practice)
                        if (canTrade2 && balance2 == 50000m && !string.IsNullOrEmpty(accountName2) && !accountName2.Contains("PRAC"))
                        {
                            tradingCombineAccount = account;
                            _logger.LogInformation("🎯 FOUND TRADING COMBINE ACCOUNT: Balance=${Balance:F2}, Name={Name}", balance2, accountName2);
                            break;
                        }
                        
                        // Fallback: any tradable account
                        if (canTrade2 && tradableAccount == null)
                        {
                            tradableAccount = account;
                        }
                    }
                    
                    // Select Trading Combine account if found, otherwise first tradable account
                    var selectedAccount = tradingCombineAccount ?? tradableAccount ?? accountsElement[0];
                    var accountSelectionReason = tradingCombineAccount.HasValue ? "Trading Combine (50K)" : 
                                                tradableAccount.HasValue ? "First Tradable" : "First Available";
                    
                    if (selectedAccount.TryGetProperty("id", out var selectedIdProp))
                    {
                        AccountId = selectedIdProp.GetInt64().ToString();
                        CurrentCredentials!.AccountId = AccountId;
                        Environment.SetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID", AccountId);
                        
                        var accountName = selectedAccount.TryGetProperty("name", out var selectedNameProp) ? selectedNameProp.GetString() : "Unknown";
                        var canTrade = selectedAccount.TryGetProperty("canTrade", out var selectedCanTradeProp) ? selectedCanTradeProp.GetBoolean() : false;
                        var balance = selectedAccount.TryGetProperty("balance", out var selectedBalProp) ? selectedBalProp.GetDecimal() : 0m;
                        
                        _logger.LogInformation("✅ SELECTED ACCOUNT ({Reason}) - ID: {AccountId}, Name: {AccountName}, CanTrade: {CanTrade}, Balance: ${Balance:F2}", 
                            accountSelectionReason, AccountId, accountName, canTrade, balance);
                        
                        // CRITICAL: Retry SignalR subscriptions with the retrieved account ID
                        if (_signalRConnectionManager != null)
                        {
                            _logger.LogInformation("🔄 Retrying SignalR subscriptions with account ID: {AccountId}", AccountId);
                            try
                            {
                                var subscriptionSuccess = await _signalRConnectionManager.RetrySubscriptionsWithAccountId(AccountId);
                                if (subscriptionSuccess)
                                {
                                    _logger.LogInformation("✅ SignalR subscriptions successful for account {AccountId}", AccountId);
                                }
                                else
                                {
                                    _logger.LogWarning("⚠️ SignalR subscriptions failed for account {AccountId}", AccountId);
                                }
                            }
                            catch (Exception ex)
                            {
                                _logger.LogWarning(ex, "⚠️ Error retrying SignalR subscriptions for account {AccountId}", AccountId);
                            }
                        }
                    }
                    else
                    {
                        _logger.LogWarning("⚠️ Account search response missing 'id' property");
                    }
                }
                else
                {
                    _logger.LogWarning("⚠️ Account search response missing or empty 'accounts' array");
                }
            }
            else
            {
                var errorContent = await searchResponse.Content.ReadAsStringAsync(cancellationToken);
                _logger.LogWarning("⚠️ Failed to search for account info - {StatusCode}: {Content}", 
                    searchResponse.StatusCode, errorContent);
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
