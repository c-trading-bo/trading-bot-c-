using System;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Helper class to probe contract entitlements via REST snapshot before attempting SignalR subscriptions
/// </summary>
public static class ContractEntitlementProbe
{
    /// <summary>
    /// Probes a contract ID via REST snapshot to verify entitlements before SignalR subscription
    /// </summary>
    /// <param name="contractId">Contract ID to probe (e.g., CON.F.US.EP.Z25)</param>
    /// <param name="jwt">JWT token for authentication</param>
    /// <param name="log">Logger instance</param>
    /// <returns>True if contract is accessible and has market data, false otherwise</returns>
    public static async Task<bool> ProbeSnapshotAsync(string contractId, string jwt, ILogger log)
    {
        using var http = new HttpClient { BaseAddress = new Uri("https://api.topstepx.com") };
        http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
        var url = $"/api/Quote/snapshot?contractId={Uri.EscapeDataString(contractId)}";
        
        try 
        {
            using var resp = await http.GetAsync(url).ConfigureAwait(false);
            var body = await resp.Content.ReadAsStringAsync().ConfigureAwait(false);
            
            if (!resp.IsSuccessStatusCode) 
            {
                log.LogWarning("[SNAPSHOT] {cid} HTTP {code} body={body}", contractId, (int)resp.StatusCode, body);
                return false;
            }
            
            // Crude "has prices" check - look for market data indicators
            if (body.Contains("last") || body.Contains("bid") || body.Contains("ask") || 
                body.Contains("price") || body.Contains("quote"))
            {
                log.LogInformation("[SNAPSHOT] {cid} OK (prices present)", contractId);
                return true;
            }
            
            log.LogWarning("[SNAPSHOT] {cid} Empty/unknown payload", contractId);
            return false;
        } 
        catch (Exception ex) 
        {
            log.LogError(ex, "[SNAPSHOT] {cid} failed", contractId);
            return false;
        }
    }
    
    /// <summary>
    /// Resolves standardized contract IDs from environment variables
    /// </summary>
    /// <returns>Tuple of (ES contract, NQ contract)</returns>
    public static (string es, string nq) ResolveContracts()
    {
        var es = Environment.GetEnvironmentVariable("TOPSTEPX_EVAL_ES_ID") ?? "CON.F.US.EP.Z25";
        var nq = Environment.GetEnvironmentVariable("TOPSTEPX_EVAL_NQ_ID") ?? "CON.F.US.ENQ.Z25"; // enforce ENQ
        return (es, nq);
    }
}