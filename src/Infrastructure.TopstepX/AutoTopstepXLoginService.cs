using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TradingBot.Abstractions;
using Infrastructure.TopstepX;
using System.Text.Json;
using System.Net.Http.Json;
using System.Linq;
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
            // Ensure BaseAddress is set
            if (_httpClient.BaseAddress == null)
            {
                _httpClient.BaseAddress = new Uri("https://api.topstepx.com");
            }
            
            _httpClient.DefaultRequestHeaders.Authorization = 
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", JwtToken);

            _logger.LogInformation("🔍 Searching for account information via TopstepX API...");
            
            // First, search for accounts to get the GUID
            var searchRequest = new
            {
                accountNumber = CurrentCredentials!.Username // Account display number
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
                    // Use intelligent account selection rules
                    var selectedAccount = SelectBestAccount(accountsElement);
                    
                    if (selectedAccount != null && selectedAccount.Value.TryGetProperty("id", out var idProp))
                    {
                        AccountId = idProp.GetInt64().ToString();
                        CurrentCredentials!.AccountId = AccountId;
                        Environment.SetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID", AccountId);
                        
                        var accountName = selectedAccount.Value.TryGetProperty("name", out var nameProp) ? nameProp.GetString() ?? "Unknown" : "Unknown";
                        var canTrade = selectedAccount.Value.TryGetProperty("canTrade", out var canTradeProp) ? canTradeProp.GetBoolean() : false;
                        var balance = selectedAccount.Value.TryGetProperty("balance", out var balProp) ? balProp.GetDecimal() : 0m;
                        var displayNumber = selectedAccount.Value.TryGetProperty("displayNumber", out var displayProp) ? displayProp.GetString() ?? "" : "";
                        var alias = selectedAccount.Value.TryGetProperty("alias", out var aliasProp) ? aliasProp.GetString() ?? "" : "";
                        var phase = selectedAccount.Value.TryGetProperty("phase", out var phaseProp) ? phaseProp.GetString() ?? "" : "";
                        var status = selectedAccount.Value.TryGetProperty("status", out var statusProp) ? statusProp.GetString() ?? "" : "";
                        
                        _logger.LogInformation("✅ Selected account: id={AccountId} display={DisplayNumber} alias={Alias} canTrade={CanTrade} phase={Phase} status={Status} balance=${Balance:F2}", 
                            AccountId, displayNumber, alias, canTrade, phase, status, balance);
                        
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
                        _logger.LogWarning("⚠️ No suitable account found in search response");
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

    /// <summary>
    /// Intelligent account selection using environment hints and business rules
    /// </summary>
    private JsonElement? SelectBestAccount(JsonElement accountsArray)
    {
        var accounts = accountsArray.EnumerateArray().ToList();
        if (!accounts.Any()) return null;

        _logger.LogInformation("📋 Found {Count} accounts, applying selection rules...", accounts.Count);

        // Environment hint overrides
        var hintDisplayNumber = Environment.GetEnvironmentVariable("TOPSTEP_ACCOUNT_DISPLAY");
        var hintAlias = Environment.GetEnvironmentVariable("TOPSTEP_ACCOUNT_ALIAS");

        // Log all accounts for visibility
        foreach (var account in accounts)
        {
            var id = account.TryGetProperty("id", out var idProp) ? idProp.GetInt64().ToString() : "?";
            var displayNumber = account.TryGetProperty("displayNumber", out var displayProp) ? displayProp.GetString() : "";
            var alias = account.TryGetProperty("alias", out var aliasProp) ? aliasProp.GetString() : "";
            var canTrade = account.TryGetProperty("canTrade", out var canTradeProp) ? canTradeProp.GetBoolean() : false;
            var status = account.TryGetProperty("status", out var statusProp) ? statusProp.GetString() : "";
            var phase = account.TryGetProperty("phase", out var phaseProp) ? phaseProp.GetString() : "";
            
            _logger.LogDebug("   Account: id={Id} display={Display} alias={Alias} canTrade={CanTrade} status={Status} phase={Phase}",
                id, displayNumber, alias, canTrade, status, phase);
        }

        // Rule 1: Environment hint for display number
        if (!string.IsNullOrEmpty(hintDisplayNumber))
        {
            foreach (var account in accounts)
            {
                if (account.TryGetProperty("displayNumber", out var dp) && 
                    dp.GetString()?.Equals(hintDisplayNumber, StringComparison.OrdinalIgnoreCase) == true)
                {
                    _logger.LogInformation("🎯 Selected account via TOPSTEP_ACCOUNT_DISPLAY hint: {DisplayNumber}", hintDisplayNumber);
                    return account;
                }
            }
        }

        // Rule 2: Environment hint for alias
        if (!string.IsNullOrEmpty(hintAlias))
        {
            foreach (var account in accounts)
            {
                if (account.TryGetProperty("alias", out var ap) && 
                    ap.GetString()?.Contains(hintAlias, StringComparison.OrdinalIgnoreCase) == true)
                {
                    _logger.LogInformation("🎯 Selected account via TOPSTEP_ACCOUNT_ALIAS hint: {Alias}", hintAlias);
                    return account;
                }
            }
        }

        // Rule 3: Active status + canTrade + Evaluation phase
        foreach (var account in accounts)
        {
            if (account.TryGetProperty("status", out var statusProp) && statusProp.GetString()?.Equals("Active", StringComparison.OrdinalIgnoreCase) == true &&
                account.TryGetProperty("canTrade", out var canTradeProp) && canTradeProp.GetBoolean() &&
                account.TryGetProperty("phase", out var phaseProp) && 
                (phaseProp.GetString()?.Contains("Eval", StringComparison.OrdinalIgnoreCase) == true ||
                 phaseProp.GetString()?.Contains("Combine", StringComparison.OrdinalIgnoreCase) == true))
            {
                _logger.LogInformation("🎯 Selected active evaluation/combine account");
                return account;
            }
        }

        // Rule 4: Active status + canTrade (any account)
        foreach (var account in accounts)
        {
            if (account.TryGetProperty("status", out var statusProp) && statusProp.GetString()?.Equals("Active", StringComparison.OrdinalIgnoreCase) == true &&
                account.TryGetProperty("canTrade", out var canTradeProp) && canTradeProp.GetBoolean())
            {
                _logger.LogInformation("🎯 Selected active tradable account");
                return account;
            }
        }

        // Rule 5: First account that can trade
        foreach (var account in accounts)
        {
            if (account.TryGetProperty("canTrade", out var canTradeProp) && canTradeProp.GetBoolean())
            {
                _logger.LogInformation("🎯 Selected first tradable account");
                return account;
            }
        }

        // Rule 6: Fallback to first account
        _logger.LogWarning("⚠️ No tradable accounts found, using first available account");
        return accounts.First();
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
