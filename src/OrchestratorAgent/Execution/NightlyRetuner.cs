using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Execution;

internal sealed class NightlyRetuner(ILogger log, HttpClient http, Func<Task<string>> getJwt, IReadOnlyDictionary<string, string> contractIdsByRoot, IEnumerable<string> roots, CancellationToken ct) : IAsyncDisposable
{
    private readonly ILogger _log = log;
    private readonly HttpClient _http = http;
    private readonly Func<Task<string>> _getJwt = getJwt;
    private readonly IReadOnlyDictionary<string, string> _contractIdsByRoot = new Dictionary<string, string>(contractIdsByRoot, StringComparer.OrdinalIgnoreCase);
    private readonly IReadOnlyList<string> _roots = [.. roots];
    private readonly CancellationToken _ct = ct;
    private readonly CancellationTokenSource _cts = new();
    private Task? _loop;

    public void Start()
    {
        if (_loop != null) return;
        _loop = Task.Run(async () =>
        {
            // Schedule: run once soon if past due, then daily at configured UTC hour
            int hourUtc = 1; // 01:00 UTC by default
            var rawHour = Environment.GetEnvironmentVariable("RETUNE_START_HOUR_UTC");
            if (!string.IsNullOrWhiteSpace(rawHour) && int.TryParse(rawHour, out var h) && h >= 0 && h <= 23) hourUtc = h;
            int days = 10; var rawDays = Environment.GetEnvironmentVariable("RETUNE_DAYS"); if (!string.IsNullOrWhiteSpace(rawDays) && int.TryParse(rawDays, out var d) && d > 0) days = d;
            var wnd = TimeSpan.FromDays(days);

            while (!_cts.IsCancellationRequested && !_ct.IsCancellationRequested)
            {
                var now = DateTime.UtcNow;
                var next = new DateTime(now.Year, now.Month, now.Day, hourUtc, 0, 0, DateTimeKind.Utc);
                if (now > next) next = next.AddDays(1);
                var delay = next - now;
                try { await Task.Delay(delay, CancellationTokenSource.CreateLinkedTokenSource(_cts.Token, _ct).Token).ConfigureAwait(false); } catch { }
                if (_cts.IsCancellationRequested || _ct.IsCancellationRequested) break;

                try
                {
                    _log.LogInformation("[Retune] Running nightly retune (lookback={Days}d) for roots: {Roots}", days, string.Join(",", _roots));
                    foreach (var root in _roots)
                    {
                        if (!_contractIdsByRoot.TryGetValue(root, out var cid) || string.IsNullOrWhiteSpace(cid)) { _log.LogWarning("[Retune] Missing contractId for {Root}", root); continue; }
                        var end = DateTime.UtcNow;
                        var start = end - wnd;
                        // S2/S3/S6/S11 — each is best-effort; errors do not abort the loop
                        try { await TuningRunner.RunS2Async(_http, _getJwt, cid, root, start, end, _log, _cts.Token).ConfigureAwait(false); } catch (Exception ex) { _log.LogWarning(ex, "[Retune] S2 {Root}", root); }
                        try { await TuningRunner.RunS3Async(_http, _getJwt, cid, root, start, end, _log, _cts.Token).ConfigureAwait(false); } catch (Exception ex) { _log.LogWarning(ex, "[Retune] S3 {Root}", root); }
                        try { await TuningRunner.RunS6Async(_http, _getJwt, cid, root, start, end, _log, _cts.Token).ConfigureAwait(false); } catch (Exception ex) { _log.LogWarning(ex, "[Retune] S6 {Root}", root); }
                        try { await TuningRunner.RunS11Async(_http, _getJwt, cid, root, start, end, _log, _cts.Token).ConfigureAwait(false); } catch (Exception ex) { _log.LogWarning(ex, "[Retune] S11 {Root}", root); }
                    }
                }
                catch (Exception ex)
                {
                    _log.LogWarning(ex, "[Retune] Nightly run failed");
                }
            }
        }, _cts.Token);
    }

    public async ValueTask DisposeAsync()
    {
        try { await _cts.CancelAsync().ConfigureAwait(false); } catch { }
        try { if (_loop != null) await _loop.ConfigureAwait(false); } catch { }
        _cts.Dispose();
    }
}
