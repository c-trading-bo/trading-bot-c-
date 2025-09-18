using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Globalization;

namespace OrchestratorAgent.Infra;

public sealed class AutoRollbackGuard : IAsyncDisposable
{
    private readonly ILogger _log;
    private readonly PositionTracker _pos;
    private readonly string _paramsDir = null!;
    private readonly TimeSpan _window;
    private readonly decimal _dropUsd;
    private readonly bool _allowLive;
    private readonly CancellationTokenSource _cts = new();
    private Task? _loop;

    private sealed record ApplyState(DateTime appliedUtc, decimal baselineRealized);
    private readonly ConcurrentDictionary<string, ApplyState> _state = new(StringComparer.OrdinalIgnoreCase);

    public AutoRollbackGuard(ILogger log, PositionTracker pos, TimeSpan window, decimal dropUsd, bool allowLive)
    {
        _log = log; _pos = pos; _window = window <= TimeSpan.Zero ? TimeSpan.FromHours(1) : window; _dropUsd = dropUsd; _allowLive = allowLive;
        _paramsDir = Path.Combine(AppContext.BaseDirectory, "state", "params");
        Directory.CreateDirectory(_paramsDir);
    }

    public void Start()
    {
        if (_loop != null) return;
        _loop = Task.Run(async () =>
        {
            _log.LogInformation("[Rollback] Guard running (window={Win} drop=${Drop})", (int)_window.TotalMinutes, _dropUsd);
            while (!_cts.IsCancellationRequested)
            {
                try { EvaluateOnce(); } catch (Exception ex) { _log.LogWarning(ex, "[Rollback] evaluate"); }
                try { await Task.Delay(TimeSpan.FromMinutes(5), _cts.Token).ConfigureAwait(false); } catch (OperationCanceledException) { /* shutdown */ }
            }
        }, _cts.Token);
    }

    private void EvaluateOnce()
    {
        if (!_allowLive)
        {
            var live = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (live) { _log.LogDebug("[Rollback] Live mode — skipping"); return; }
        }
        if (_dropUsd <= 0) return;

        // Load latest apply events from history.jsonl (best-effort)
        var history = Path.Combine(_paramsDir, "history.jsonl");
        var map = new Dictionary<string, (DateTime appliedUtc, string strat, string root)>();
        try
        {
            if (File.Exists(history))
            {
                foreach (var line in File.ReadLines(history))
                {
                    try
                    {
                        if (string.IsNullOrWhiteSpace(line)) continue;
                        var doc = JsonDocument.Parse(line);
                        var rootEl = doc.RootElement;
                        var evt = TryStr(rootEl, "evt");
                        if (!string.Equals(evt, "apply", StringComparison.OrdinalIgnoreCase)) continue;
                        var strat = TryStr(rootEl, "strat") ?? "";
                        var root = TryStr(rootEl, "root") ?? "";
                        var ts = TryDt(rootEl, "ts") ?? DateTime.UtcNow;
                        if (!string.IsNullOrWhiteSpace(strat) && !string.IsNullOrWhiteSpace(root))
                            map[$"{strat}:{root}"] = (ts, strat, root);
                    }
                    catch (Exception)
                    {
                        // ignore malformed history lines
                    }
                }
            }
        }
        catch (Exception)
        {
            // best-effort history read
        }

        // Track new applies, set baseline realized
        foreach (var (key, tup) in map)
        {
            _state.AddOrUpdate(key,
                _ => new ApplyState(tup.appliedUtc, RealizedForRoot(tup.root)),
                (_, prev) => prev);
        }

        // Evaluate active applies within window
        var now = DateTime.UtcNow;
        foreach (var kv in _state.ToArray())
        {
            var key = kv.Key; var st = kv.Value;
            if (now - st.appliedUtc > _window) { _state.TryRemove(key, out _); continue; }
            var parts = key.Split(':'); if (parts.Length != 2) continue;
            var strat = parts[0]; var root = parts[1];
            var realized = RealizedForRoot(root);
            var delta = realized - st.baselineRealized;
            if (delta <= -_dropUsd)
            {
                // Revert: delete override file and log history
                var file = Path.Combine(_paramsDir, $"{strat}-{root}.json");
                try
                {
                    if (File.Exists(file)) File.Delete(file);
                    // Minimal history line
                    AppendHistory("revert", strat, root, new { reason = "degrade", delta, window_min = (int)_window.TotalMinutes });
                    _log.LogWarning("[Rollback] Reverted {Key} due to degrade delta=${Delta:F2}", key, delta);

                    // Canary penalty: subtract delta from the arm that was running (if known)
                    try
                    {
                        var canaryPath = Path.Combine(_paramsDir, "canary.json");
                        if (File.Exists(canaryPath))
                        {
                            var text = File.ReadAllText(canaryPath);
                            var json = System.Text.Json.JsonDocument.Parse(text).RootElement;
                            if (json.ValueKind == JsonValueKind.Object)
                            {
                                var keyReg = strat + ":" + (json.TryGetProperty("lastRegime", out var lr) ? (lr.GetString() ?? "range") : "range");
                                // best-effort: just bump a generic "range_cons" arm for this key (format-agnostic)
                                var newObj = new System.Text.Json.Nodes.JsonObject();
                                foreach (var prop in json.EnumerateObject()) newObj[prop.Name] = System.Text.Json.Nodes.JsonNode.Parse(prop.Value.GetRawText());
                                var statsNode = newObj["stats"] as System.Text.Json.Nodes.JsonObject ?? [];
                                var armsNode = statsNode[keyReg] as System.Text.Json.Nodes.JsonObject ?? [];
                                var armKey = "range_cons";
                                var armNode = armsNode[armKey] as System.Text.Json.Nodes.JsonObject ?? [];
                                int plays = (int?)armNode["plays"] ?? 0; decimal total = (decimal?)armNode["totalReward"] ?? 0m;
                                armNode["plays"] = plays + 1; armNode["totalReward"] = total + delta; // delta is negative
                                armsNode[armKey] = armNode; statsNode[keyReg] = armsNode; newObj["stats"] = statsNode;
                                File.WriteAllText(canaryPath, newObj.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
                            }
                        }
                    }
                    catch (Exception) { /* best-effort canary penalty */ }
                }
                catch (Exception ex) { _log.LogWarning(ex, "[Rollback] Failed to revert {Key}", key); }
                finally { _state.TryRemove(key, out _); }
            }
        }
    }

    private decimal RealizedForRoot(string root)
    {
        try
        {
            var snap = _pos.Snapshot();
            decimal sum = 0m;
            foreach (var kv in snap)
            {
                try
                {
                    var k = kv.Key;
                    var r = OrchestratorAgent.SymbolMeta.RootFromName(k);
                    if (string.Equals(r, root, StringComparison.OrdinalIgnoreCase))
                        sum += kv.Value.RealizedUsd;
                }
                catch (Exception)
                {
                    // ignore symbol parse errors
                }
            }
            return sum;
        }
        catch (Exception)
        {
            return 0m;
        }
    }

    private static string? TryStr(JsonElement e, string name)
    {
        try { if (e.TryGetProperty(name, out var p) && p.ValueKind == JsonValueKind.String) return p.GetString(); }
        catch (Exception) { /* ignore */ }
        return null;
    }
    private static DateTime? TryDt(JsonElement e, string name)
    {
        try
        {
            if (e.TryGetProperty(name, out var p) && p.ValueKind == JsonValueKind.String)
            {
                var s = p.GetString();
                if (DateTime.TryParse(s, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var dt)) return dt;
            }
        }
        catch (Exception) { /* ignore parse */ }
        return null;
    }

    private void AppendHistory(string evt, string strat, string root, object payload)
    {
        try
        {
            var path = Path.Combine(_paramsDir, "history.jsonl");
            var rec = new { ts = DateTime.UtcNow, evt, strat, root, payload };
            File.AppendAllText(path, JsonSerializer.Serialize(rec) + Environment.NewLine);
        }
        catch (Exception) { /* best-effort */ }
    }

    public async ValueTask DisposeAsync()
    {
        try { await _cts.CancelAsync().ConfigureAwait(false); } catch (Exception) { /* best-effort cancel */ }
        try { if (_loop != null) await _loop.ConfigureAwait(false); } catch (Exception) { /* ignore */ }
        _cts.Dispose();
    }
}
