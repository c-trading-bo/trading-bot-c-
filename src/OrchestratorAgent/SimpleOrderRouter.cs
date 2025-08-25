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
            }\r

            try
            {
                var token = await _getJwtAsync();
                if (string.IsNullOrWhiteSpace(token))
                {
                    _log.LogWarning("[Router] Missing JWT; cannot place order.");
                    await JournalAsync(false, "missing_jwt", 0, null, modeStr);
                    return false;
                }

                var placeBody = new
                {
                    accountId = sig.AccountId,
                    contractId = sig.ContractId,
                    type = 1, // 1 = Limit
                    side = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase) ? 1 : 0,
                    size = sig.Size > 0 ? sig.Size : 1,
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

        private static string Trunc(string? s, int max = 256)
        {
            if (string.IsNullOrEmpty(s)) return string.Empty;
            return s.Length <= max ? s : s.Substring(0, max) + "…";
        }
    }
}
