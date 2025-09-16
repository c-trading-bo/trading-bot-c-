using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// AccountService with real /api/Account GET + periodic refresh
/// NO STUBS - Uses actual TopstepX account API for live portfolio data  
/// </summary>
public interface IAccountService
{
    Task<AccountInfo> GetAccountInfoAsync();
    Task<PositionInfo[]> GetPositionsAsync();
    Task<decimal> GetAccountBalanceAsync();
    Task StartPeriodicRefreshAsync(TimeSpan interval);
    event Action<AccountInfo> OnAccountUpdated;
    
    // Additional methods needed by RealTopstepXClient
    Task<AccountInfo?> GetAccountAsync(string accountId, CancellationToken cancellationToken);
    Task<BalanceInfo?> GetAccountBalanceAsync(string accountId, CancellationToken cancellationToken);
    Task<PositionInfo[]?> GetAccountPositionsAsync(string accountId, CancellationToken cancellationToken);
    Task<AccountInfo[]?> SearchAccountsAsync(CancellationToken cancellationToken);
}

public record BalanceInfo(
    decimal CurrentBalance,
    decimal AvailableBalance,
    decimal BuyingPower,
    decimal DayPnL,
    decimal UnrealizedPnL,
    decimal Equity = 0m,
    decimal TotalValue = 0m,
    decimal MarginUsed = 0m,
    decimal FreeMargin = 0m,
    string Currency = "USD",
    decimal RiskPercentage = 0m
);

public record AccountInfo(
    string AccountId,
    decimal Balance,
    decimal BuyingPower,
    decimal DayPnL,
    decimal UnrealizedPnL,
    string Status,
    string Type = "Funded",
    decimal Equity = 0m,
    bool IsActive = true,
    string RiskLevel = "Medium",
    DateTime LastUpdated = default
);

public record PositionInfo(
    string Symbol,
    int Quantity,
    decimal AvgPrice,
    decimal MarketValue,
    decimal UnrealizedPnL,
    string Side = "Long",
    decimal AveragePrice = 0m,
    decimal MarketPrice = 0m,
    decimal RealizedPnL = 0m,
    decimal NetValue = 0m,
    DateTime OpenTime = default,
    DateTime LastUpdated = default,
    decimal RiskAmount = 0m,
    decimal MarginRequirement = 0m
);

public class AccountService : IAccountService, IDisposable
{
    private readonly ILogger<AccountService> _logger;
    private readonly AppOptions _config;
    private readonly HttpClient _httpClient;
    private readonly Timer _refreshTimer;
    
    public event Action<AccountInfo>? OnAccountUpdated;

    public AccountService(ILogger<AccountService> logger, IOptions<AppOptions> config, HttpClient httpClient)
    {
        _logger = logger;
        _config = config.Value;
        _httpClient = httpClient;
        _httpClient.BaseAddress = new Uri(_config.ApiBase);
        
        // Create timer but don't start it yet
        _refreshTimer = new Timer(async _ => await RefreshAccountAsync(), null, Timeout.Infinite, Timeout.Infinite);
    }

    public async Task<AccountInfo> GetAccountInfoAsync()
    {
        const int maxRetries = 3;
        
        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                // Real GET to /api/Account/{accountId}
                var response = await _httpClient.GetAsync($"/api/Account/{_config.AccountId}");
                
                if (response.IsSuccessStatusCode)
                {
                    var json = await response.Content.ReadAsStringAsync();
                    var accountData = JsonSerializer.Deserialize<JsonElement>(json);
                    
                    var accountInfo = new AccountInfo(
                        _config.AccountId,
                        accountData.GetProperty("balance").GetDecimal(),
                        accountData.GetProperty("buyingPower").GetDecimal(),
                        accountData.GetProperty("dayPnL").GetDecimal(),
                        accountData.GetProperty("unrealizedPnL").GetDecimal(),
                        accountData.GetProperty("status").GetString() ?? "Unknown"
                    );

                    _logger.LogInformation("[ACCOUNT] Balance: {Balance:C}, Day P&L: {DayPnL:C}, Status: {Status}", 
                        accountInfo.Balance, accountInfo.DayPnL, accountInfo.Status);

                    return accountInfo;
                }
                else if (ShouldRetry(response.StatusCode) && attempt < maxRetries)
                {
                    _logger.LogWarning("[ACCOUNT] Attempt {Attempt}/{Max} failed: HTTP {StatusCode}, retrying...", 
                        attempt, maxRetries, (int)response.StatusCode);
                    await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)));
                    continue;
                }
                else
                {
                    // Don't retry 4xx errors or final attempt
                    response.EnsureSuccessStatusCode();
                }
            }
            catch (HttpRequestException ex) when (attempt < maxRetries)
            {
                _logger.LogWarning(ex, "[ACCOUNT] HTTP request failed on attempt {Attempt}/{Max}, retrying...", 
                    attempt, maxRetries);
                await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)));
            }
        }

        // This should not be reached due to EnsureSuccessStatusCode above
        throw new InvalidOperationException("Failed to get account info after all retry attempts");
    }

    public async Task<PositionInfo[]> GetPositionsAsync()
    {
        const int maxRetries = 3;
        
        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                var response = await _httpClient.GetAsync($"/api/Account/{_config.AccountId}/positions");
                
                if (response.IsSuccessStatusCode)
                {
                    var json = await response.Content.ReadAsStringAsync();
                    var positionsArray = JsonSerializer.Deserialize<JsonElement[]>(json);
                    
                    var positions = positionsArray?.Select(p => new PositionInfo(
                        p.GetProperty("symbol").GetString() ?? "",
                        p.GetProperty("quantity").GetInt32(),
                        p.GetProperty("avgPrice").GetDecimal(),
                        p.GetProperty("marketValue").GetDecimal(),
                        p.GetProperty("unrealizedPnL").GetDecimal()
                    )).ToArray() ?? Array.Empty<PositionInfo>();

                    _logger.LogInformation("[ACCOUNT] Retrieved {Count} positions", positions.Length);
                    return positions;
                }
                else if (ShouldRetry(response.StatusCode) && attempt < maxRetries)
                {
                    _logger.LogWarning("[ACCOUNT] Positions request attempt {Attempt}/{Max} failed: HTTP {StatusCode}, retrying...", 
                        attempt, maxRetries, (int)response.StatusCode);
                    await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)));
                    continue;
                }
                else
                {
                    // Don't retry 4xx errors or final attempt
                    response.EnsureSuccessStatusCode();
                }
            }
            catch (HttpRequestException ex) when (attempt < maxRetries)
            {
                _logger.LogWarning(ex, "[ACCOUNT] Positions HTTP request failed on attempt {Attempt}/{Max}, retrying...", 
                    attempt, maxRetries);
                await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)));
            }
        }

        // This should not be reached due to EnsureSuccessStatusCode above
        throw new InvalidOperationException("Failed to get positions after all retry attempts");
    }

    public async Task<decimal> GetAccountBalanceAsync()
    {
        var accountInfo = await GetAccountInfoAsync();
        return accountInfo.Balance;
    }

    public async Task StartPeriodicRefreshAsync(TimeSpan interval)
    {
        _logger.LogInformation("[ACCOUNT] Starting periodic refresh every {Interval}", interval);
        
        // Start the timer to refresh account data periodically
        _refreshTimer.Change(TimeSpan.Zero, interval);
        
        // Perform initial account data validation
        await ValidateAccountConfigurationAsync();
        
        // Log startup completion
        _logger.LogInformation("[ACCOUNT] Account refresh service started with {Interval} interval", interval);
    }

    private async Task RefreshAccountAsync()
    {
        try
        {
            var accountInfo = await GetAccountInfoAsync();
            OnAccountUpdated?.Invoke(accountInfo);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ACCOUNT] Error during periodic refresh");
        }
    }

    /// <summary>
    /// Determine if HTTP status code should trigger a retry (5xx/408 only)
    /// </summary>
    private static bool ShouldRetry(System.Net.HttpStatusCode statusCode)
    {
        return statusCode == System.Net.HttpStatusCode.RequestTimeout || // 408
               statusCode == System.Net.HttpStatusCode.InternalServerError || // 500
               statusCode == System.Net.HttpStatusCode.BadGateway || // 502
               statusCode == System.Net.HttpStatusCode.ServiceUnavailable || // 503
               statusCode == System.Net.HttpStatusCode.GatewayTimeout; // 504
    }

    public void Dispose()
    {
        _refreshTimer?.Dispose();
    }

    /// <summary>
    /// Validate account configuration and connectivity
    /// </summary>
    private async Task ValidateAccountConfigurationAsync()
    {
        try
        {
            // Validate environment configuration
            var apiBase = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE");
            if (string.IsNullOrEmpty(apiBase))
            {
                _logger.LogWarning("[ACCOUNT] TOPSTEPX_API_BASE not configured");
                return;
            }

            // Test basic connectivity
            using var httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
            var healthCheckUrl = $"{apiBase.TrimEnd('/')}/health";
            
            try
            {
                var response = await httpClient.GetAsync(healthCheckUrl);
                if (response.IsSuccessStatusCode)
                {
                    _logger.LogInformation("[ACCOUNT] API connectivity validated successfully");
                    _logger.LogInformation("[ACCOUNT] API connectivity validation passed");
                }
                else
                {
                    _logger.LogWarning("[ACCOUNT] API health check returned: {StatusCode}", response.StatusCode);
                }
            }
            catch (TaskCanceledException)
            {
                _logger.LogWarning("[ACCOUNT] API connectivity test timed out");
            }
            catch (HttpRequestException ex)
            {
                _logger.LogWarning(ex, "[ACCOUNT] API connectivity test failed");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ACCOUNT] Error during account configuration validation");
        }
    }
    
    // Implementation of additional methods needed by RealTopstepXClient
    public async Task<AccountInfo?> GetAccountAsync(string accountId, CancellationToken cancellationToken)
    {
        try
        {
            var accountInfo = await GetAccountInfoAsync();
            if (accountInfo.AccountId == accountId)
            {
                return accountInfo;
            }
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ACCOUNT] Failed to get account {AccountId}", accountId);
            return null;
        }
    }
    
    public async Task<BalanceInfo?> GetAccountBalanceAsync(string accountId, CancellationToken cancellationToken)
    {
        try
        {
            var accountInfo = await GetAccountInfoAsync();
            if (accountInfo.AccountId == accountId)
            {
                return new BalanceInfo(
                    accountInfo.Balance,
                    accountInfo.BuyingPower,
                    accountInfo.BuyingPower,
                    accountInfo.DayPnL,
                    accountInfo.UnrealizedPnL
                );
            }
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ACCOUNT] Failed to get balance for account {AccountId}", accountId);
            return null;
        }
    }
    
    public async Task<PositionInfo[]?> GetAccountPositionsAsync(string accountId, CancellationToken cancellationToken)
    {
        try
        {
            var positions = await GetPositionsAsync();
            return positions;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ACCOUNT] Failed to get positions for account {AccountId}", accountId);
            return null;
        }
    }
    
    public async Task<AccountInfo[]?> SearchAccountsAsync(CancellationToken cancellationToken)
    {
        try
        {
            var accountInfo = await GetAccountInfoAsync();
            return new[] { accountInfo };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ACCOUNT] Failed to search accounts");
            return null;
        }
    }
}