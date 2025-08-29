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
        private readonly OrchestratorAgent.Ops.PartialExitService? _partialExit;
        private volatile bool _entriesDisabled = false;

        public SimpleOrderRouter(HttpClient http, Func<Task<string?>> getJwtAsync, ILogger log, bool live)
        {
            _http = http;
            _getJwtAsync = getJwtAsync;
            _log = log;
            _live = live;
            _partialExit = null;
        }

        public SimpleOrderRouter(HttpClient http, Func<Task<string?>> getJwtAsync, ILogger log, bool live, OrchestratorAgent.Ops.PartialExitService? partialExit)
        {
            _http = http;
            _getJwtAsync = getJwtAsync;
            _log = log;
            _live = live;
            _partialExit = partialExit;
        }

        public void DisableAllEntries() => _entriesDisabled = true;
        public void EnableAllEntries() => _entriesDisabled = false;
        public async Task CloseAll(string reason, CancellationToken ct)
        {
            try
            {
                _log.LogInformation("[Router] CloseAll requested: reason={Reason}", reason);
                // Best-effort stub: try to resolve JWT and enumerate open positions, then log.
                var token = await _getJwtAsync();
                if (string.IsNullOrWhiteSpace(token)) { _log.LogWarning("[Router] CloseAll: missing JWT"); return; }
                using var req = new HttpRequestMessage(HttpMethod.Post, "/api/Position/searchOpen");
                req.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
                // If server requires accountId, backend may infer from JWT; leave body minimal.
                req.Content = new StringContent("{}", Encoding.UTF8, "application/json");
                using var resp = await _http.SendAsync(req, ct);
                var txt = await resp.Content.ReadAsStringAsync(ct);
                _log.LogInformation("[Router] CloseAll ack: status={Status} body.len={Len}", (int)resp.StatusCode, txt?.Length ?? 0);
                // Intentionally not placing market exits here due to missing account context; relying on higher-level supervisor if present.
            }
            catch (Exception ex)
            {
                try { _log.LogWarning(ex, "[Router] CloseAll failed"); } catch { }
            }
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

            // Enforce ES/MES rounding and validate risk > 0 when stop present
            decimal tick = BotCore.Models.InstrumentMeta.Tick(sig.Symbol);
            if (tick <= 0) tick = 0.25m;
            static decimal RoundToTick(decimal px, decimal t) => t <= 0 ? px : Math.Round(px / t, 0, MidpointRounding.AwayFromZero) * t;
            var roundedEntry = sig.Entry > 0 ? RoundToTick(sig.Entry, tick) : sig.Entry;
            var roundedStop = sig.Stop > 0 ? RoundToTick(sig.Stop, tick) : sig.Stop;
            var roundedT1 = sig.Target > 0 ? RoundToTick(sig.Target, tick) : sig.Target;
            if (roundedEntry != sig.Entry || roundedStop != sig.Stop || roundedT1 != sig.Target)
            {
                sig = sig with { Entry = roundedEntry, Stop = roundedStop, Target = roundedT1 };
            }

            if (_entriesDisabled)
            {
                _log.LogInformation("[Router] Entries disabled — blocking order (tag={Tag}).", sig.Tag);
                await JournalAsync(false, "entries_disabled", 0, null, modeStr);
                return false;
            }

            // Enforce global size cap (max 2) for safety
            sig = sig with { Size = Math.Clamp(sig.Size, 1, 2) };

            // Ensure a deterministic order_group_id tag for idempotency if none provided
            if (string.IsNullOrWhiteSpace(sig.Tag))
            {
                // Use strategy|symbol|entry|stop rounded to tick as the idempotent tag basis
                var tag = $"{sig.StrategyId}|{sig.Symbol}|{sig.Entry:F2}|{sig.Stop:F2}".ToUpperInvariant();
                sig = sig with { Tag = tag };
            }

            // Risk sanity: require positive risk when a stop is provided
            if (sig.Stop != 0 && sig.Entry != 0)
            {
                var isShort = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase);
                var risk = isShort ? (sig.Stop - sig.Entry) : (sig.Entry - sig.Stop);
                if (risk <= 0)
                {
                    _log.LogInformation("[Router] risk<=0 — blocking: entry={Entry} stop={Stop} side={Side}", sig.Entry, sig.Stop, sig.Side);
                    await JournalAsync(false, "risk_non_positive", 0, null, modeStr);
                    return false;
                }
            }

            // Structured signal line per spec
            try
            {
                var isBuy = string.Equals(sig.Side, "BUY", StringComparison.OrdinalIgnoreCase) || string.Equals(sig.Side, "LONG", StringComparison.OrdinalIgnoreCase);
                var line = new
                {
                    sig = sig.StrategyId,
                    side = isBuy ? "BUY" : "SELL",
                    symbol = sig.Symbol,
                    qty = sig.Size,
                    entry = Math.Round(sig.Entry, 2),
                    stop = Math.Round(sig.Stop, 2),
                    t1 = Math.Round(sig.Target, 2),
                    tag = sig.Tag,
                    mode = modeStr
                };
                Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(line));
            }
            catch { }
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

                // Idempotent routing: search for an existing open parent with same customTag first
                try
                {
                    using var searchReq = new HttpRequestMessage(HttpMethod.Post, "/api/Order/search");
                    searchReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                    var body = new { accountId = sig.AccountId, contractId = sig.ContractId, status = "OPEN", customTag = sig.Tag };
                    searchReq.Content = new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json");
                    using var searchResp = await _http.SendAsync(searchReq, ct);
                    var sBody = await searchResp.Content.ReadAsStringAsync(ct);
                    if (searchResp.IsSuccessStatusCode && !string.IsNullOrWhiteSpace(sBody) && sBody.Contains(sig.Tag ?? string.Empty, StringComparison.OrdinalIgnoreCase))
                    {
                        _log.LogInformation("[Router] Idempotent: found existing open order for tag={Tag}; skip new place.", sig.Tag);
                        await JournalAsync(true, "idempotent_skip", (int)searchResp.StatusCode, null, modeStr);
                        return true;
                    }
                }
                catch { }

                // Enforce per-order cap and global net position cap (max 2)
                int maxContracts = 2;
                try { var raw = Environment.GetEnvironmentVariable("MAX_CONTRACTS"); if (int.TryParse(raw, out var mc) && mc > 0) maxContracts = mc; } catch { }
                var requested = sig.Size > 0 ? sig.Size : 1;
                var clampedSize = Math.Min(requested, maxContracts);
                if (clampedSize != requested)
                    _log.LogInformation("[Router] Clamped size from {Req} to {Clamp} (MAX_CONTRACTS={Max})", requested, clampedSize, maxContracts);

                // Net position guard: ensure resulting net does not exceed ±1
                async Task<int> GetOpenNetQtyAsync()
                {
                    try
                    {
                        using var posReq = new HttpRequestMessage(HttpMethod.Post, "/api/Position/searchOpen");
                        posReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                        var body = new { accountId = sig.AccountId };
                        posReq.Content = new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json");
                        using var posResp = await _http.SendAsync(posReq, ct);
                        var pText = await posResp.Content.ReadAsStringAsync(ct);
                        if (!posResp.IsSuccessStatusCode || string.IsNullOrWhiteSpace(pText)) return 0;
                        using var doc = JsonDocument.Parse(pText);
                        int sum = 0;
                        if (doc.RootElement.TryGetProperty("positions", out var arr) && arr.ValueKind == JsonValueKind.Array)
                        {
                            foreach (var el in arr.EnumerateArray())
                            {
                                try
                                {
                                    if (el.TryGetProperty("size", out var sz))
                                    {
                                        if (sz.ValueKind == JsonValueKind.Number && sz.TryGetInt32(out var n)) sum += n;
                                        else if (sz.ValueKind == JsonValueKind.String && int.TryParse(sz.GetString(), out var ns)) sum += ns;
                                    }
                                }
                                catch { }
                            }
                        }
                        return sum;
                    }
                    catch { return 0; }
                }

                var currentNet = await GetOpenNetQtyAsync();
                var sgn = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase) ? -1 : 1;
                int netCap = 1;
                int absCur = Math.Abs(currentNet);
                int maxAdd = (currentNet == 0 || Math.Sign(currentNet) == sgn)
                    ? Math.Max(0, netCap - absCur) // same direction: top up to cap
                    : Math.Min(clampedSize, netCap + absCur); // opposite: allow flatten and flip up to cap
                var finalSize = Math.Min(clampedSize, maxAdd);
                if (finalSize <= 0)
                {
                    _log.LogInformation("[Router] Net cap: blocking order — current={Cur} side={Side} req={Req} would exceed ±1", currentNet, sig.Side, requested);
                    await JournalAsync(false, "net_cap", 0, null, modeStr);
                    return false;
                }

                var placeBody = new
                {
                    accountId = sig.AccountId,
                    contractId = sig.ContractId,
                    type = 1, // 1 = Limit
                    side = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase) ? 1 : 0,
                    size = finalSize,
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
                    // R-driven bracket math with env fallbacks
                    // tick already computed above
                    int AbsTicks(decimal a) => (int)Math.Max(1, Math.Round(Math.Abs(a) / tick));
                    bool isSell = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase);

                    int slTicks = 0;
                    if (sig.Stop != 0 && sig.Entry != 0)
                        slTicks = AbsTicks(sig.Entry - sig.Stop); // long: entry-stop; short distance still positive

                    int EnvInt(string key, int defVal) { try { var raw = Environment.GetEnvironmentVariable(key); return int.TryParse(raw, out var v) && v > 0 ? v : defVal; } catch { return defVal; } }
                    decimal EnvDec(string key, decimal defVal) { try { var raw = Environment.GetEnvironmentVariable(key); return decimal.TryParse(raw, out var v) ? v : defVal; } catch { return defVal; } }

                    var rTp1 = EnvDec("R_MULT_TP1", 1.0m);
                    var rTp2 = EnvDec("R_MULT_TP2", 2.0m);
                    var beR = EnvDec("BE_AFTER_R", 1.0m);
                    var tp1Pct = EnvDec("TP1_PCT_CLOSE", 0.50m);
                    int trailTicks = EnvInt("BRACKET_TRAIL_TICKS", 6);

                    // Fallbacks if stop not provided
                    if (slTicks <= 0) slTicks = EnvInt($"BRACKET_SL_{sig.Symbol.ToUpperInvariant()}_TICKS", EnvInt("BRACKET_SL_TICKS", 12));
                    int tpTicks = slTicks > 0 ? (int)Math.Round(slTicks * rTp2) : EnvInt($"BRACKET_TP_{sig.Symbol.ToUpperInvariant()}_TICKS", EnvInt("BRACKET_TP_TICKS", 20));
                    int beTicks = slTicks > 0 ? (int)Math.Round(slTicks * beR) : EnvInt("BRACKET_BE_TICKS", 8);
                    if (tpTicks <= 0) tpTicks = EnvInt($"BRACKET_TP_{sig.Symbol.ToUpperInvariant()}_TICKS", EnvInt("BRACKET_TP_TICKS", 20));
                    if (beTicks <= 0) beTicks = EnvInt("BRACKET_BE_TICKS", 8);

                    // Upsert platform bracket (acts as TP2)
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
                        var stopPx = isSell ? sig.Entry + slTicks * tick : sig.Entry - slTicks * tick;
                        var tgtPx = isSell ? sig.Entry - tpTicks * tick : sig.Entry + tpTicks * tick;
                        _log.LogInformation("STOP NEW   {Stop}  ({Sl} ticks)", stopPx, slTicks);
                        _log.LogInformation("TARGET NEW {Tgt}  ({Tp} ticks)", tgtPx, tpTicks);
                    }
                    else
                    {
                        _log.LogWarning("[Router] Bracket attach failed for parent {ParentId}", parentId);
                    }

                    // Schedule TP1 partial (reduce-only IOC) at 1R
                    try
                    {
                        int tp1Ticks = slTicks > 0 ? (int)Math.Round(slTicks * rTp1) : 0;
                        if (tp1Ticks > 0 && _partialExit != null)
                        {
                            var lot = BotCore.Models.InstrumentMeta.LotStep(sig.Symbol);
                            int tp1Qty = (int)Math.Max(0, Math.Floor(clampedSize * (double)tp1Pct));
                            if (lot > 1) tp1Qty -= (tp1Qty % lot);
                            if (tp1Qty > 0)
                            {
                                _ = Task.Run(() => _partialExit!.TryScaleOutAsync(sig.Symbol, parentId!, clampedSize, sig.Entry, !isSell, tp1Ticks, tp1Qty, ct), ct);
                                _log.LogInformation("[TP1] scheduled reduce-only partial qty={Qty} at {Ticks} ticks ({R}R)", tp1Qty, tp1Ticks, rTp1);
                            }
                        }
                    }
                    catch { }

                    // Entry TTL with single smart reprice then cancel
                    _ = Task.Run(async () =>
                    {
                        try
                        {
                            var t0 = DateTime.UtcNow; var ttl = TimeSpan.FromSeconds(int.TryParse(Environment.GetEnvironmentVariable("ENTRY_TTL_SEC"), out var tsec) && tsec > 0 ? tsec : 45);
                            var repriceAt = TimeSpan.FromSeconds(int.TryParse(Environment.GetEnvironmentVariable("ENTRY_REPRICE_AT_SEC"), out var rp) && rp > 0 ? rp : 25);
                            var didReprice = false;
                            while (DateTime.UtcNow - t0 < ttl && !ct.IsCancellationRequested)
                            {
                                try { await Task.Delay(1000, ct); } catch { break; }
                                try
                                {
                                    using var getReq = new HttpRequestMessage(HttpMethod.Get, $"/orders/{parentId}");
                                    getReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                                    using var getResp = await _http.SendAsync(getReq, ct);
                                    var body = await getResp.Content.ReadAsStringAsync(ct);
                                    if (!getResp.IsSuccessStatusCode) continue;
                                    using var jdoc = JsonDocument.Parse(string.IsNullOrWhiteSpace(body) ? "{}" : body);
                                    var root = jdoc.RootElement;
                                    int qty = 0, filled = 0; decimal price = sig.Entry;
                                    try { if (root.TryGetProperty("qty", out var q)) qty = q.GetInt32(); } catch { }
                                    try { if (root.TryGetProperty("filledQty", out var f)) filled = f.GetInt32(); } catch { }
                                    try { if (root.TryGetProperty("price", out var p)) price = p.GetDecimal(); else if (root.TryGetProperty("limitPrice", out var lp)) price = lp.GetDecimal(); } catch { }
                                    if (filled >= qty && qty > 0) break; // fully filled
                                    if (!didReprice && DateTime.UtcNow - t0 > repriceAt)
                                    {
                                        var newPx = (string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase)) ? price - tick : price + tick;
                                        newPx = RoundToTick(newPx, tick);
                                        try
                                        {
                                            using var modReq = new HttpRequestMessage(HttpMethod.Post, "/orders/modify");
                                            modReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                                            modReq.Content = new StringContent(JsonSerializer.Serialize(new { orderId = parentId, price = newPx }), Encoding.UTF8, "application/json");
                                            using var modResp = await _http.SendAsync(modReq, ct);
                                            if (!modResp.IsSuccessStatusCode)
                                            {
                                                using var modReq2 = new HttpRequestMessage(HttpMethod.Post, "/api/Order/modify");
                                                modReq2.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                                                modReq2.Content = new StringContent(JsonSerializer.Serialize(new { orderId = parentId, price = newPx }), Encoding.UTF8, "application/json");
                                                await _http.SendAsync(modReq2, ct);
                                            }
                                            _log.LogInformation("[TTL] Repriced parent {ParentId} -> {Px}", parentId, newPx);
                                            didReprice = true;
                                        }
                                        catch { }
                                    }
                                }
                                catch { }
                            }
                            if (DateTime.UtcNow - t0 >= ttl)
                            {
                                try
                                {
                                    using var cancelReq = new HttpRequestMessage(HttpMethod.Post, "/orders/cancel");
                                    cancelReq.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                                    cancelReq.Content = new StringContent(JsonSerializer.Serialize(new { orderId = parentId }), Encoding.UTF8, "application/json");
                                    using var canResp = await _http.SendAsync(cancelReq, ct);
                                    if (!canResp.IsSuccessStatusCode)
                                    {
                                        using var cancelReq2 = new HttpRequestMessage(HttpMethod.Post, "/api/Order/cancel");
                                        cancelReq2.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                                        cancelReq2.Content = new StringContent(JsonSerializer.Serialize(new { orderId = parentId }), Encoding.UTF8, "application/json");
                                        await _http.SendAsync(cancelReq2, ct);
                                    }
                                    _log.LogWarning("[TTL] Canceled unfilled parent {ParentId} after TTL", parentId);
                                }
                                catch { }
                            }
                        }
                        catch { }
                    }, ct);
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

        public async Task FlattenAll(long accountId, CancellationToken ct)
        {
            try
            {
                var token = await _getJwtAsync();
                if (string.IsNullOrWhiteSpace(token)) return;
                using var req = new HttpRequestMessage(HttpMethod.Post, "/positions/flatten_all");
                req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                req.Content = new StringContent(JsonSerializer.Serialize(new { accountId }), Encoding.UTF8, "application/json");
                using var resp = await _http.SendAsync(req, ct);
                var body = await resp.Content.ReadAsStringAsync(ct);
                _log.LogInformation("[Router] FlattenAll posted: {Status} {BodyLen}", (int)resp.StatusCode, (body?.Length ?? 0));
            }
            catch (Exception ex)
            {
                try { _log.LogWarning(ex, "[Router] FlattenAll failed"); } catch { }
            }
        }

        private static string Trunc(string? s, int max = 256)
        {
            if (string.IsNullOrEmpty(s)) return string.Empty;
            return s.Length <= max ? s : s[..max] + "…";
        }
    }
}
