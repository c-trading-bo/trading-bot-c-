#nullable enable
using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using BotCore;
using SupervisorAgent;

namespace OrchestratorAgent.Health
{
    public sealed class Preflight(ApiClient api, StatusService status, Preflight.TradingProfileConfig cfg, long accountId)
    {
        private readonly ApiClient _api = api;
        private readonly StatusService _status = status;
        private readonly long _accountId = accountId;
        private readonly TradingProfileConfig _cfg = cfg;
        private readonly DateTimeOffset _startUtc = DateTimeOffset.UtcNow;

        public sealed class TradingProfileConfig
        {
            public RiskConfig Risk { get; set; } = new();
            public sealed class RiskConfig
            {
                public decimal DailyLossLimit { get; set; } = 1000m;
                public int MaxTradesPerDay { get; set; } = 1000;
            }
        }

        public async Task<(bool ok, string msg)> RunAsync(string rootSymbol, CancellationToken ct)
        {
            // 1) Auth fresh (T-120s)
            static string? Env(string name) => Environment.GetEnvironmentVariable(name);
            var jwt = Env("TOPSTEPX_JWT") ?? Env("JWT") ?? string.Empty;
            if (string.IsNullOrWhiteSpace(jwt)) return (false, "JWT missing");
            try
            {
                if (IsJwtExpiring(jwt, TimeSpan.FromSeconds(120))) return (false, "JWT expiring");
            }
            catch (Exception ex)
            {
                // Log JWT parsing error but continue with best-effort validation
                Console.WriteLine($"Warning: JWT parsing failed: {ex.Message}");
            }

            // 2) Hubs connected (via StatusService snapshot)
            var u = _status.Get<string>("user.state");
            var m = _status.Get<string>("market.state");
            if (string.IsNullOrWhiteSpace(m)) return (false, "MarketHub disconnected");
            if (string.IsNullOrWhiteSpace(u)) return (false, "UserHub disconnected");

            // 2.5) Resolve active contractId for the root (e.g., "ES" -> "CON.F.US.EP.Z25")
            string contractId;
            try
            {
                if (_status.Contracts != null && _status.Contracts.TryGetValue(rootSymbol, out var id) && !string.IsNullOrWhiteSpace(id))
                {
                    contractId = id;
                }
                else
                {
                    contractId = await _api.ResolveContractIdAsync(rootSymbol, ct).ConfigureAwait(false).ConfigureAwait(false);
                    try 
                    { 
                        (_status.Contracts ??= [])[rootSymbol] = contractId; 
                    } 
                    catch (Exception ex) 
                    { 
                        Console.WriteLine($"Warning: Failed to cache contract ID: {ex.Message}"); 
                    }
                }
            }
            catch
            {
                return (false, "Contract resolve failed");
            }

            // 3) Quotes & bars freshness: prefer vendor UTC timestamp (lastUpdated) with safe fallbacks
            var now = DateTimeOffset.UtcNow;
            var lastQuoteUpdated = _status.Get<DateTimeOffset?>($"last.quote.updated.{contractId}")
                                 ?? _status.Get<DateTimeOffset?>("last.quote.updated");
            var lastQuoteIngest = _status.Get<DateTimeOffset?>($"last.quote.{contractId}")
                           ?? _status.Get<DateTimeOffset?>("last.quote");
            var lastBarIngest = _status.Get<DateTimeOffset?>($"last.bar.{contractId}")
                         ?? _status.Get<DateTimeOffset?>("last.bar");

            // Warm-up: allow up to 30s on cold start before requiring quotes
            var uptime = now - _startUtc;
            if (!lastQuoteUpdated.HasValue && !lastQuoteIngest.HasValue)
            {
                if (uptime < TimeSpan.FromSeconds(30))
                    return (false, $"Warming up quotes ({(int)uptime.TotalSeconds}s)");
                return (false, "No quotes yet");
            }

            var quoteAge = lastQuoteUpdated.HasValue ? now - lastQuoteUpdated.Value
                         : lastQuoteIngest.HasValue ? now - lastQuoteIngest.Value
                         : TimeSpan.MaxValue;
            var barAge = lastBarIngest.HasValue ? now - lastBarIngest.Value : TimeSpan.MaxValue;

            if (quoteAge > TimeSpan.FromSeconds(5)) return (false, $"Quotes stale ({(int)quoteAge.TotalSeconds}s)");
            if (barAge > TimeSpan.FromSeconds(30)) return (false, $"Bars stale ({(int)barAge.TotalSeconds}s)");

            // 4) Risk counters (PnL & trades) – best-effort against two possible endpoints
            decimal? netPnl = null;
            try
            {
                var pnlA = await _api.GetAsync<JsonElement>($"/accounts/{_accountId}/pnl?scope=today", ct).ConfigureAwait(false).ConfigureAwait(false);
                if (pnlA.ValueKind != JsonValueKind.Undefined)
                {
                    if (pnlA.TryGetProperty("net", out var netProp) && netProp.TryGetDecimal(out var net)) netPnl = net;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to retrieve primary P&L data: {ex.Message}");
            }
            if (netPnl is null)
            {
                try
                {
                    var pnlB = await _api.GetAsync<JsonElement>($"/api/Account/pnl?accountId={_accountId}&scope=today", ct).ConfigureAwait(false).ConfigureAwait(false);
                    if (pnlB.ValueKind != JsonValueKind.Undefined)
                    {
                        if (pnlB.TryGetProperty("net", out var netProp) && netProp.TryGetDecimal(out var net)) netPnl = net;
                        else if (pnlB.TryGetProperty("realized", out var r) && pnlB.TryGetProperty("commissions", out var c) && pnlB.TryGetProperty("fees", out var f))
                        {
                            decimal rr = r.TryGetDecimal(out var rv) ? rv : 0m;
                            decimal cc = c.TryGetDecimal(out var cv) ? cv : 0m;
                            decimal ff = f.TryGetDecimal(out var fv) ? fv : 0m;
                            netPnl = rr - (cc + ff);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Failed to retrieve fallback P&L data: {ex.Message}");
                }
            }
            if (netPnl.HasValue && netPnl.Value <= -_cfg.Risk.DailyLossLimit)
                return (false, "Daily loss tripped");

            int? tradesCount = null;
            try
            {
                var trades = await _api.GetAsync<JsonElement>($"/accounts/{_accountId}/trades?scope=today", ct).ConfigureAwait(false).ConfigureAwait(false);
                if (trades.ValueKind == JsonValueKind.Array) tradesCount = trades.GetArrayLength();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to retrieve primary trades data: {ex.Message}");
            }
            if (tradesCount is null)
            {
                try
                {
                    var trades = await _api.GetAsync<JsonElement>($"/api/Order/search?accountId={_accountId}&status=FILLED&scope=today", ct).ConfigureAwait(false).ConfigureAwait(false);
                    if (trades.ValueKind == JsonValueKind.Array) tradesCount = trades.GetArrayLength();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Failed to retrieve fallback trades data: {ex.Message}");
                }
            }
            if (tradesCount.HasValue && tradesCount.Value >= _cfg.Risk.MaxTradesPerDay)
                return (false, "Max trades reached");

            // 5) Rollover sanity – best-effort endpoint
            try
            {
                var fm = await _api.GetAsync<JsonElement>($"/contracts/resolve_front?symbol={rootSymbol}", ct).ConfigureAwait(false).ConfigureAwait(false);
                if (fm.ValueKind == JsonValueKind.Object)
                {
                    bool expSoon = fm.TryGetProperty("isExpiringSoon", out var e) && e.GetBoolean();
                    bool rolled = fm.TryGetProperty("rolled", out var r) && r.GetBoolean();
                    if (expSoon && !rolled) return (false, "Rollover pending");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Failed to check rollover status: {ex.Message}");
            }

            return (true, "OK");
        }

        private static bool IsJwtExpiring(string jwt, TimeSpan before)
        {
            // Decode JWT payload and read `exp` (Unix seconds)
            var parts = jwt.Split('.');
            if (parts.Length < 2) return true;
            var payload = parts[1];
            var pad = 4 - (payload.Length % 4);
            if (pad is > 0 and < 4) payload += new string('=', pad);
            payload = payload.Replace('-', '+').Replace('_', '/');
            var bytes = Convert.FromBase64String(payload);
            var json = JsonDocument.Parse(bytes);
            if (!json.RootElement.TryGetProperty("exp", out var expEl)) return true;
            var exp = DateTimeOffset.FromUnixTimeSeconds(expEl.GetInt64());
            return DateTimeOffset.UtcNow >= exp - before;
        }
    }
}
