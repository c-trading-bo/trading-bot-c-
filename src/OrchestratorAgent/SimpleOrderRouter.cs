using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Models;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent
{
    internal sealed class SimpleOrderRouter
    {
        private readonly HttpClient _http;
        private readonly Func<Task<string?>> _getJwtAsync;
        private readonly ILogger _log;
        private readonly bool _live;

        public SimpleOrderRouter(HttpClient http, Func<Task<string?>> getJwtAsync, ILogger log, bool live)
        {
            _http = http;
            _getJwtAsync = getJwtAsync;
            _log = log;
            _live = live;
        }

        public async Task<bool> RouteAsync(Signal sig, CancellationToken ct)
        {
            if (sig is null) return false;
            if (sig.AccountId <= 0 || string.IsNullOrWhiteSpace(sig.ContractId))
            {
                _log.LogWarning("[Router] Missing accountId/contractId for signal {Tag}; skipping.", sig.Tag);
                await JournalAsync(false, "missing_account_or_contract", 0, null, "N/A");
                return false;
            }

            bool EnvFlag(string key)
            {
                var raw = Environment.GetEnvironmentVariable(key);
                if (string.IsNullOrWhiteSpace(raw)) return false;
                raw = raw.Trim();
                return raw.Equals("1", StringComparison.OrdinalIgnoreCase) || raw.Equals("true", StringComparison.OrdinalIgnoreCase) || raw.Equals("yes", StringComparison.OrdinalIgnoreCase);
            }

            var kill = EnvFlag("KILL_SWITCH");
            var liveEnv = EnvFlag("LIVE_ORDERS");
            var liveMode = (_live || liveEnv) && !kill;
            var modeStr = liveMode ? "LIVE" : "DRY-RUN";

            _log.LogInformation("[Router] {Mode} Route: {Side} {Size} {Contract} @ {Entry} (stop {Stop}, target {Target}) tag={Tag}",
                modeStr, sig.Side, sig.Size, sig.ContractId, sig.Entry, sig.Stop, sig.Target, sig.Tag);

            if (kill)
            {
                _log.LogWarning("[Router] KILL_SWITCH active — blocking order.");
                await JournalAsync(false, "kill_switch", 0, null, modeStr);
                return false;
            }

            if (!liveMode)
            {
                await JournalAsync(true, "dry_run", 0, null, modeStr);
                return true;
            }

            try
            {
                var token = await _getJwtAsync();
                if (string.IsNullOrWhiteSpace(token))
                {
                    _log.LogWarning("[Router] Missing JWT; cannot place order.");
                    await JournalAsync(false, "missing_jwt", 0, null, modeStr);
                    return false;
                }

                // Enforce max contracts cap (default 2; override via MAX_CONTRACTS env)
                int maxContracts = 2;
                try { var raw = Environment.GetEnvironmentVariable("MAX_CONTRACTS"); if (int.TryParse(raw, out var mc) && mc > 0) maxContracts = mc; } catch { }
                var requested = sig.Size > 0 ? sig.Size : 1;
                var clampedSize = Math.Min(requested, maxContracts);
                if (clampedSize != requested)
                    _log.LogInformation("[Router] Clamped size from {Req} to {Clamp} (MAX_CONTRACTS={Max})", requested, clampedSize, maxContracts);

                var placeBody = new
                {
                    accountId = sig.AccountId,
                    contractId = sig.ContractId,
                    type = 1, // 1 = Limit
                    side = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase) ? 1 : 0,
                    size = clampedSize,
                    limitPrice = sig.Entry,
                    customTag = sig.Tag
                };

                using var req = new HttpRequestMessage(HttpMethod.Post, "/api/Order/place");
                req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                req.Content = new StringContent(JsonSerializer.Serialize(placeBody), Encoding.UTF8, "application/json");

                using var resp = await _http.SendAsync(req, ct);
                var text = await resp.Content.ReadAsStringAsync(ct);
                if (!resp.IsSuccessStatusCode)
                {
                    _log.LogWarning("[Router] Place failed {Status}: {Body}", (int)resp.StatusCode, Trunc(text));
                    await JournalAsync(false, "http_fail", (int)resp.StatusCode, text, modeStr);
                    return false;
                }

                _log.LogInformation("[Router] Place OK: {Body}", Trunc(text));
                await JournalAsync(true, "ok", (int)resp.StatusCode, text, modeStr);

                // Parse parent orderId
                string? parentId = null;
                try
                {
                    using var doc = JsonDocument.Parse(string.IsNullOrWhiteSpace(text) ? "{}" : text);
                    var root = doc.RootElement;
                    if (root.ValueKind == JsonValueKind.Object)
                    {
                        if (root.TryGetProperty("orderId", out var oid)) parentId = oid.ToString();
                        else if (root.TryGetProperty("id", out var idEl)) parentId = idEl.ToString();
                    }
                }
                catch { }

                if (!string.IsNullOrWhiteSpace(parentId))
                {
                    // Compute ticks from provided prices or fall back to env defaults
                    decimal tick = BotCore.Models.InstrumentMeta.Tick(sig.Symbol);
                    int AbsTicks(decimal a) => (int)Math.Max(1, Math.Round(Math.Abs(a) / (tick <= 0 ? 0.25m : tick)));
                    bool isSell = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase);
                    int tpTicks = 0;
                    int slTicks = 0;
                    if (sig.Target != 0 && sig.Entry != 0)
                    {
                        tpTicks = AbsTicks(sig.Target - sig.Entry);
                    }
                    if (sig.Stop != 0 && sig.Entry != 0)
                    {
                        slTicks = AbsTicks(sig.Entry - sig.Stop); // long: entry-stop; short will still be positive distance
                    }
                    // Environment overrides (global and per symbol)
                    int EnvInt(string key, int defVal)
                    {
                        try { var raw = Environment.GetEnvironmentVariable(key); return int.TryParse(raw, out var v) && v > 0 ? v : defVal; } catch { return defVal; }
                    }
                    int beTicks = EnvInt("BRACKET_BE_TICKS", 8);
                    int trailTicks = EnvInt("BRACKET_TRAIL_TICKS", 6);
                    if (tpTicks <= 0) tpTicks = EnvInt($"BRACKET_TP_{sig.Symbol.ToUpperInvariant()}_TICKS", EnvInt("BRACKET_TP_TICKS", 20));
                    if (slTicks <= 0) slTicks = EnvInt($"BRACKET_SL_{sig.Symbol.ToUpperInvariant()}_TICKS", EnvInt("BRACKET_SL_TICKS", 12));

                    // Try primary upsert endpoint first
                    var upsertBody = new { parentId = (object?)parentId, qty = clampedSize, takeProfitTicks = tpTicks, stopLossTicks = slTicks, breakevenAfterTicks = beTicks, trailTicks };
                    bool bracketsOk = false;
                    try
                    {
                        using var upReq = new HttpRequestMessage(HttpMethod.Post, "/orders/brackets/upsert");
                        upReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                        upReq.Content = new StringContent(JsonSerializer.Serialize(upsertBody), Encoding.UTF8, "application/json");
                        using var upResp = await _http.SendAsync(upReq, ct);
                        if (upResp.IsSuccessStatusCode) bracketsOk = true;
                    }
                    catch { }

                    if (!bracketsOk)
                    {
                        try
                        {
                            var body2 = new { parentId = (object?)parentId, takeProfitTicks = tpTicks, stopLossTicks = slTicks };
                            using var req2 = new HttpRequestMessage(HttpMethod.Post, "/orders/brackets");
                            req2.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                            req2.Content = new StringContent(JsonSerializer.Serialize(body2), Encoding.UTF8, "application/json");
                            using var resp2 = await _http.SendAsync(req2, ct);
                            bracketsOk = resp2.IsSuccessStatusCode;
                        }
                        catch { }
                    }
                    if (!bracketsOk)
                    {
                        try
                        {
                            var body3 = new { orderId = (object?)parentId, takeProfitTicks = tpTicks, stopLossTicks = slTicks };
                            using var req3 = new HttpRequestMessage(HttpMethod.Post, "/api/Order/brackets/upsert");
                            req3.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                            req3.Content = new StringContent(JsonSerializer.Serialize(body3), Encoding.UTF8, "application/json");
                            using var resp3 = await _http.SendAsync(req3, ct);
                            bracketsOk = resp3.IsSuccessStatusCode;
                        }
                        catch { }
                    }

                    if (bracketsOk)
                    {
                        // Narrative logs per your diagram
                        var stopPx = isSell
                            ? sig.Entry + slTicks * (tick <= 0 ? 0.25m : tick)
                            : sig.Entry - slTicks * (tick <= 0 ? 0.25m : tick);
                        var tgtPx = isSell
                            ? sig.Entry - tpTicks * (tick <= 0 ? 0.25m : tick)
                            : sig.Entry + tpTicks * (tick <= 0 ? 0.25m : tick);
                        _log.LogInformation("STOP NEW   {Stop}  ({Sl} ticks)", stopPx, slTicks);
                        _log.LogInformation("TARGET NEW {Tgt}  ({Tp} ticks)", tgtPx, tpTicks);
                    }
                    else
                    {
                        _log.LogWarning("[Router] Bracket attach failed for parent {ParentId}", parentId);
                    }
                }
                return true;
            }
            catch (OperationCanceledException)
            {
                await JournalAsync(false, "canceled", 0, null, modeStr);
                return false;
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[Router] Unexpected error placing order.");
                await JournalAsync(false, ex.GetType().Name, 0, ex.Message, modeStr);
                return false;
            }

            async Task JournalAsync(bool success, string? reason, int status, string? body, string mode)
            {
                try
                {
                    var dir = Environment.GetEnvironmentVariable("JOURNAL_DIR") ?? "journal";
                    System.IO.Directory.CreateDirectory(dir);
                    var path = System.IO.Path.Combine(dir, "orders.jsonl");
                    var line = JsonSerializer.Serialize(new
                    {
                        ts = DateTimeOffset.UtcNow,
                        mode,
                        success,
                        reason,
                        status,
                        signal = new
                        {
                            sig.StrategyId,
                            sig.Symbol,
                            sig.Side,
                            sig.Entry,
                            sig.Stop,
                            sig.Target,
                            sig.Size,
                            sig.AccountId,
                            sig.ContractId,
                            sig.Tag
                        },
                        body = string.IsNullOrEmpty(body) ? null : Trunc(body, 512)
                    });
                    await System.IO.File.AppendAllTextAsync(path, line + Environment.NewLine, ct);
                }
                catch { }
            }
        }

        public async Task EnsureBracketsAsync(long accountId, CancellationToken ct)
        {
            try
            {
                var token = await _getJwtAsync();
                if (string.IsNullOrWhiteSpace(token)) return;
                using var req = new HttpRequestMessage(HttpMethod.Get, $"/orders?accountId={accountId}&status=OPEN&parent=true");
                req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                using var resp = await _http.SendAsync(req, ct);
                var body = await resp.Content.ReadAsStringAsync(ct);
                if (!resp.IsSuccessStatusCode) return;
                using var doc = JsonDocument.Parse(string.IsNullOrWhiteSpace(body) ? "[]" : body);
                if (doc.RootElement.ValueKind != JsonValueKind.Array) return;
                int attached = 0;
                foreach (var el in doc.RootElement.EnumerateArray())
                {
                    bool hasBr = false;
                    string? id = null;
                    try
                    {
                        if (el.ValueKind == JsonValueKind.Object)
                        {
                            if (el.TryGetProperty("hasBrackets", out var hb) && hb.ValueKind == JsonValueKind.True) hasBr = true;
                            if (el.TryGetProperty("id", out var idEl)) id = idEl.ToString();
                        }
                    }
                    catch { }
                    if (hasBr || string.IsNullOrWhiteSpace(id)) continue;
                    try
                    {
                        int tp = 20, sl = 12;
                        var body2 = new { parentId = (object?)id, takeProfitTicks = tp, stopLossTicks = sl };
                        using var req2 = new HttpRequestMessage(HttpMethod.Post, "/orders/brackets");
                        req2.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                        req2.Content = new StringContent(JsonSerializer.Serialize(body2), Encoding.UTF8, "application/json");
                        using var resp2 = await _http.SendAsync(req2, ct);
                        if (resp2.IsSuccessStatusCode) attached++;
                    }
                    catch { }
                }
                if (attached > 0) _log.LogInformation("[Router] EnsureBrackets attached for {Count} parent(s)", attached);
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[Router] EnsureBracketsAsync error");
            }
        }

        private static string Trunc(string? s, int max = 256)
        {
            if (string.IsNullOrEmpty(s)) return string.Empty;
            return s.Length <= max ? s : s.Substring(0, max) + "…";
        }
    }
}
