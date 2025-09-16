using System;
using System.Net.Http;
using System.Net.NetworkInformation;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// Environment validator for TopstepX trading operations
/// Checks NTP clock sync and network connectivity at startup
/// </summary>
public interface IEnvironmentValidator
{
    Task<bool> ValidateEnvironmentAsync();
    Task<bool> CheckClockSyncAsync();
    Task<bool> CheckNetworkConnectivityAsync();
    Task<bool> CheckTopstepXConnectivityAsync();
}

public class EnvironmentValidator : IEnvironmentValidator
{
    private readonly ILogger<EnvironmentValidator> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly HttpClient _httpClient;

    public EnvironmentValidator(
        ILogger<EnvironmentValidator> logger,
        ITradingLogger tradingLogger,
        HttpClient httpClient)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _httpClient = httpClient;
    }

    public async Task<bool> ValidateEnvironmentAsync()
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "EnvironmentValidator",
            "Starting environment validation for TopstepX trading");

        var clockSync = await CheckClockSyncAsync();
        var networkConnectivity = await CheckNetworkConnectivityAsync();
        var topstepXConnectivity = await CheckTopstepXConnectivityAsync();

        var isValid = clockSync && networkConnectivity && topstepXConnectivity;
        
        if (isValid)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "EnvironmentValidator",
                "✅ Environment validation passed - ready for trading operations");
        }
        else
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "EnvironmentValidator",
                "❌ Environment validation failed - trading operations may be unreliable");
        }

        return isValid;
    }

    public async Task<bool> CheckClockSyncAsync()
    {
        try
        {
            // Get current system time
            var systemTime = DateTime.UtcNow;
            
            // Try to get NTP time from pool.ntp.org
            var ntpTime = await GetNtpTimeAsync();
            
            if (ntpTime.HasValue)
            {
                var drift = Math.Abs((systemTime - ntpTime.Value).TotalSeconds);
                
                if (drift > 30) // More than 30 seconds drift
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "EnvironmentValidator",
                        $"⚠️ Clock drift detected: {drift:F1} seconds. System: {systemTime:HH:mm:ss}, NTP: {ntpTime.Value:HH:mm:ss}");
                    return false;
                }
                else if (drift > 5) // More than 5 seconds drift but less than 30
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "EnvironmentValidator",
                        $"Minor clock drift: {drift:F1} seconds. Consider syncing system clock.");
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "EnvironmentValidator",
                        $"✅ Clock sync validated - drift: {drift:F1} seconds");
                }
                
                return true;
            }
            else
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "EnvironmentValidator",
                    "Could not validate clock sync - NTP server unavailable, assuming valid");
                return true; // Don't fail startup if NTP is unavailable
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error checking clock sync");
            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "EnvironmentValidator",
                $"Clock sync check failed: {ex.Message}, assuming valid");
            return true; // Don't fail startup on clock check errors
        }
    }

    public async Task<bool> CheckNetworkConnectivityAsync()
    {
        try
        {
            // Check basic internet connectivity by pinging Google DNS
            var ping = new Ping();
            var reply = await ping.SendPingAsync("8.8.8.8", 5000);
            
            if (reply.Status == IPStatus.Success)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "EnvironmentValidator",
                    $"✅ Network connectivity validated - ping: {reply.RoundtripTime}ms");
                return true;
            }
            else
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "EnvironmentValidator",
                    $"❌ Network connectivity failed - status: {reply.Status}");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking network connectivity");
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "EnvironmentValidator",
                $"Network connectivity check failed: {ex.Message}");
            return false;
        }
    }

    public async Task<bool> CheckTopstepXConnectivityAsync()
    {
        try
        {
            // Check if we can reach TopstepX endpoints
            var endpoints = new[]
            {
                "https://api.topstepx.com",
                "https://rtc.topstepx.com"
            };

            var allReachable = true;
            
            foreach (var endpoint in endpoints)
            {
                try
                {
                    using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
                    var response = await _httpClient.GetAsync(endpoint, cts.Token);
                    
                    // We don't care about the HTTP status code, just that we can reach the endpoint
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "EnvironmentValidator",
                        $"✅ {endpoint} reachable - status: {response.StatusCode}");
                }
                catch (TaskCanceledException)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "EnvironmentValidator",
                        $"⚠️ {endpoint} timeout - slow network connection");
                }
                catch (HttpRequestException ex)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "EnvironmentValidator",
                        $"❌ {endpoint} unreachable: {ex.Message}");
                    allReachable = false;
                }
            }

            if (allReachable)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "EnvironmentValidator",
                    "✅ TopstepX connectivity validated - all endpoints reachable");
            }

            return allReachable;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking TopstepX connectivity");
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "EnvironmentValidator",
                $"TopstepX connectivity check failed: {ex.Message}");
            return false;
        }
    }

    private async Task<DateTime?> GetNtpTimeAsync()
    {
        try
        {
            // Simple NTP client implementation
            var ntpServer = "pool.ntp.org";
            var ntpData = new byte[48];
            ntpData[0] = 0x1B; // LI, VN, Mode

            using var udpClient = new System.Net.Sockets.UdpClient();
            
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            await udpClient.SendAsync(ntpData, ntpData.Length, ntpServer, 123);
            
            var response = await udpClient.ReceiveAsync();
            var responseData = response.Buffer;

            // Extract the seconds part from bytes 40-43
            ulong intPart = (ulong)responseData[40] << 24 |
                           (ulong)responseData[41] << 16 |
                           (ulong)responseData[42] << 8 |
                           (ulong)responseData[43];

            // Extract the fraction part from bytes 44-47
            ulong fractPart = (ulong)responseData[44] << 24 |
                             (ulong)responseData[45] << 16 |
                             (ulong)responseData[46] << 8 |
                             (ulong)responseData[47];

            var milliseconds = (intPart * 1000) + ((fractPart * 1000) / 0x100000000L);
            var networkDateTime = new DateTime(1900, 1, 1, 0, 0, 0, DateTimeKind.Utc).AddMilliseconds(milliseconds);

            return networkDateTime;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error getting NTP time");
            throw new InvalidOperationException("Failed to retrieve NTP time for clock validation", ex);
        }
    }
}