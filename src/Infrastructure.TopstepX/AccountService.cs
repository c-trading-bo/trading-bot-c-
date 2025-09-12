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
}

public record AccountInfo(
    string AccountId,
    decimal Balance,
    decimal BuyingPower,
    decimal DayPnL,
    decimal UnrealizedPnL,
    string Status
);

public record PositionInfo(
    string Symbol,
    int Quantity,
    decimal AvgPrice,
    decimal MarketValue,
    decimal UnrealizedPnL
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
        
        await Task.CompletedTask;
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
}