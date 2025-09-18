using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.IO;

namespace OrchestratorAgent.Execution;

public sealed class ContinuousRetuner(ILogger log, HttpClient http, Func<Task<string>> getJwt,
    IReadOnlyDictionary<string, string> contractIdsByRoot, IEnumerable<string> roots,
    TimeSpan interval, int lookbackDays, bool allowLive, CancellationToken ct) : IAsyncDisposable
{
    private readonly ILogger _log = log;
    private readonly HttpClient _http = http;
    private readonly Func<Task<string>> _getJwt = getJwt;
    private readonly IReadOnlyDictionary<string, string> _contractIdsByRoot = new Dictionary<string, string>(contractIdsByRoot, StringComparer.OrdinalIgnoreCase);
    private readonly IReadOnlyList<string> _roots = [.. roots];
    private readonly TimeSpan _interval = interval <= TimeSpan.Zero ? TimeSpan.FromHours(1) : interval;
    private readonly int _lookbackDays = Math.Max(1, lookbackDays);
    private readonly bool _allowLive = allowLive;
    private readonly CancellationToken _ct = ct;
    private readonly CancellationTokenSource _cts = new();
    private Task? _loop;
    private int _running;

    public void Start()
    {
        if (_loop != null) return;
        _loop = Task.Run(async () =>
        {
            while (!_cts.IsCancellationRequested && !_ct.IsCancellationRequested)
            {
                try
                {
                    if (!_allowLive)
                    {
                        var live = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (live) { _log.LogDebug("[Retune] Live mode — skipping continuous retune"); }
                        else await RunOnceAsync().ConfigureAwait(false);
                    }
                    else
                    {
                        await RunOnceAsync().ConfigureAwait(false);
                    }
                }
                catch (Exception ex)
                {
                    _log.LogWarning(ex, "[Retune] Continuous loop error");
                }
                try { await Task.Delay(_interval, CancellationTokenSource.CreateLinkedTokenSource(_cts.Token, _ct).Token).ConfigureAwait(false); } catch { }
            }
        }, _cts.Token);
    }

    private async Task RunOnceAsync()
    {
        if (System.Threading.Interlocked.Exchange(ref _running, 1) == 1)
        {
            _log.LogDebug("[Retune] Previous continuous run still in progress — skipping");
            return;
        }
        try
        {
            // Sliding 7-day (or configured) window aligned to whole UTC days
            var until = DateTime.UtcNow.Date.AddDays(1).AddTicks(-1);
            var since = until.AddDays(-_lookbackDays + 1).Date; // e.g., Mon..Mon (7 days inclusive)
            _log.LogInformation("[Retune] Continuous retune window: {Since:yyyy-MM-dd} -> {Until:yyyy-MM-dd}", since, until);

            // Ensure JWT if allowed
            try { var tok = await _getJwt().ConfigureAwait(false); if (!string.IsNullOrWhiteSpace(tok)) { /* HttpClient header set upstream */ } } catch { }

            var results = new System.Collections.Generic.List<object>();
            foreach (var root in _roots)
            {
                if (!_contractIdsByRoot.TryGetValue(root, out var cid) || string.IsNullOrWhiteSpace(cid))
                { _log.LogWarning("[Retune] Missing contractId for {Root}", root); continue; }
                results.Add(RunOne(root, "S2", async () => await TuningRunner.RunS2Async(_http, _getJwt, cid, root, since, until, _log, _cts.Token))).ConfigureAwait(false);
                results.Add(RunOne(root, "S3", async () => await TuningRunner.RunS3Async(_http, _getJwt, cid, root, since, until, _log, _cts.Token))).ConfigureAwait(false);
                results.Add(RunOne(root, "S6", async () => await TuningRunner.RunS6Async(_http, _getJwt, cid, root, since, until, _log, _cts.Token))).ConfigureAwait(false);
                results.Add(RunOne(root, "S11", async () => await TuningRunner.RunS11Async(_http, _getJwt, cid, root, since, until, _log, _cts.Token))).ConfigureAwait(false);
            }

            // Persist status for dashboard
            try
            {
                var pathDir = Path.Combine(AppContext.BaseDirectory, "state", "tuning"); Directory.CreateDirectory(pathDir);
                var status = new
                {
                    kind = "continuous",
                    nowUtc = DateTime.UtcNow,
                    lookbackDays = _lookbackDays,
                    intervalMinutes = (int)_interval.TotalMinutes,
                    since,
                    until,
                    roots = _roots,
                    results
                };
                File.WriteAllText(Path.Combine(pathDir, "retune_status.json"), JsonSerializer.Serialize(status, new JsonSerializerOptions { WriteIndented = true }));
            }
            catch { /* best-effort */ }
        }
        finally
        {
            System.Threading.Interlocked.Exchange(ref _running, 0);
        }
    }

    private static object RunOne(string root, string strat, Func<Task> run)
    {
        try { run().GetAwaiter().GetResult(); return new { root, strat, ok = true, error = (string?)null }; }
        catch (Exception ex) { return new { root, strat, ok = false, error = ex.Message }; }
    }

    public async ValueTask DisposeAsync()
    {
        try { await _cts.CancelAsync().ConfigureAwait(false); } catch { }
        try { if (_loop != null) await _loop.ConfigureAwait(false); } catch { }
        _cts.Dispose();
    }
}
