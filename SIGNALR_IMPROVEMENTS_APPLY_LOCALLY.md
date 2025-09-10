# Apply SignalR Improvements Locally

## Quick Summary
We fixed the 401 SignalR authentication errors by:
1. ✅ Removed manual `?access_token=` URL injection 
2. ✅ Implemented proper `AccessTokenProvider` pattern
3. ✅ Added JWT caching with refresh logic
4. ✅ Enhanced connection state management

## Files to Download from Codespace

1. **Main Fix**: `/workspaces/trading-bot-c-/signalr-improvements.patch`
2. **Instructions**: This file

## How to Apply Locally

### Option 1: Apply the Patch
```bash
# In your local trading-bot-c- directory
curl -o signalr-improvements.patch https://raw.githubusercontent.com/YOUR_USERNAME/trading-bot-c-/main/signalr-improvements.patch
git apply signalr-improvements.patch
```

### Option 2: Manual Copy-Paste
If the patch doesn't work, you can manually update the file:

**Target File**: `src/BotCore/Services/TradingSystemIntegrationService.cs`

**Key Changes to Make:**

1. **Add import** at top:
```csharp
using System.Net.Http.Json;
```

2. **Replace the SignalR setup section** (~line 154) with proper JWT provider:
```csharp
// User Hub Connection with proper JWT provider (no manual token in URL)
_userHubConnection = new HubConnectionBuilder()
    .WithUrl(_config.UserHubUrl, options =>
    {
        // SignalR will append ?access_token=<value> for WebSockets automatically
        options.AccessTokenProvider = async () =>
        {
            try
            {
                var token = await GetFreshJwtAsync();
                return token;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to get JWT for User Hub connection");
                return Environment.GetEnvironmentVariable("TOPSTEPX_JWT") ?? "";
            }
        };
        options.Transports = HttpTransportType.WebSockets;
    })
    .WithAutomaticReconnect(new[] { TimeSpan.Zero, TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10) })
    .Build();

// Market Hub Connection with proper JWT provider (no manual token in URL)
_marketHubConnection = new HubConnectionBuilder()
    .WithUrl(_config.MarketHubUrl, options =>
    {
        // SignalR will append ?access_token=<value> for WebSockets automatically
        options.AccessTokenProvider = async () =>
        {
            try
            {
                var token = await GetFreshJwtAsync();
                return token;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to get JWT for Market Hub connection");
                return Environment.GetEnvironmentVariable("TOPSTEPX_JWT") ?? "";
            }
        };
        options.Transports = HttpTransportType.WebSockets;
    })
    .WithAutomaticReconnect(new[] { TimeSpan.Zero, TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10) })
    .Build();
```

3. **Add the JWT management methods** at the end of the class:

```csharp
private string? _jwt;
private DateTimeOffset _jwtExpiration = DateTimeOffset.MinValue;

private async Task<string> GetFreshJwtAsync()
{
    // Check if we have a valid cached JWT (with 5-minute buffer)
    if (!string.IsNullOrEmpty(_jwt) && DateTimeOffset.UtcNow < _jwtExpiration.AddMinutes(-5))
    {
        return _jwt;
    }

    // Get fresh JWT from TopstepX
    try
    {
        _logger.LogInformation("🔄 Obtaining fresh JWT token from TopstepX...");
        
        using var client = new HttpClient();
        client.BaseAddress = new Uri("https://api.topstepx.com");
        client.DefaultRequestHeaders.Add("x-api-key", _config.ApiKey);

        var response = await client.PostAsJsonAsync("/api/Auth/authenticate", new { });
        response.EnsureSuccessStatusCode();
        
        var authResult = await response.Content.ReadFromJsonAsync<AuthResponse>();
        
        if (authResult?.access_token == null)
        {
            throw new InvalidOperationException("No access_token in auth response");
        }

        _jwt = authResult.access_token;
        _jwtExpiration = DateTimeOffset.UtcNow.AddHours(23); // TopstepX tokens last 24h, cache for 23h
        
        _logger.LogInformation("✅ Fresh JWT obtained and cached until {Expiration}", _jwtExpiration);
        return _jwt;
    }
    catch (Exception ex)
    {
        _logger.LogError(ex, "❌ Failed to obtain fresh JWT token");
        
        // Fallback to environment variable if available
        var envJwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
        if (!string.IsNullOrEmpty(envJwt))
        {
            _logger.LogWarning("🔄 Using fallback JWT from environment variable");
            return envJwt;
        }
        
        throw;
    }
}

private class AuthResponse
{
    public string? access_token { get; set; }
    public string? token_type { get; set; }
    public int expires_in { get; set; }
}
```

## Test After Applying

1. Build: `dotnet build`
2. Run: `dotnet run --project src/UnifiedOrchestrator`
3. Look for: "✅ User Hub connected successfully" and "✅ Market Hub connected successfully"

## What This Fixes

- ❌ **Before**: 401 Unauthorized on WebSocket upgrade
- ✅ **After**: Clean SignalR connections with proper JWT

The key insight: TopstepX SignalR expects `AccessTokenProvider` function, not manual URL tokens!
