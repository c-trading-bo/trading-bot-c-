#nullable enable
using System.Collections.Concurrent;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Channels;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Dashboard;

/// <summary>
/// Wire /dashboard (HTML), /data/history (GET) and /stream/realtime (SSE).
/// Feed it with Hub.OnTick(...) and a metrics snapshot delegate.
/// </summary>
public static class DashboardModule
{
    public static void MapDashboard(this WebApplication app, RealtimeHub hub)
    {
        // static file (our single-page UI)
        app.MapGet("/dashboard", async ctx =>
        {
            ctx.Response.ContentType = "text/html; charset=utf-8";
            await ctx.Response.SendFileAsync(Path.Combine(app.Environment.ContentRootPath, "wwwroot", "dashboard.html"));
        });

        // lightweight history endpoint: ?symbol=ES&res=1&from=unixSec&to=unixSec
        app.MapGet("/data/history", (string symbol, string res, long from, long to) =>
        {
            var bars = hub.QueryHistory(symbol, res, from, to);
            return Results.Json(bars);
        });

        // realtime SSE: ?symbol=ES&res=1
        app.MapGet("/stream/realtime", async (HttpContext ctx, string symbol, string res) =>
        {
            ctx.Response.Headers.CacheControl = "no-store";
            ctx.Response.Headers.Connection   = "keep-alive";
            ctx.Response.ContentType          = "text/event-stream";

            var subscription = hub.Subscribe(symbol, res);
            using var sub = subscription.sub;
            var ch = subscription.channel;
            var cancel = ctx.RequestAborted;

            // send a hello + recent last bar
            await ctx.Response.WriteAsync($"event: hello\ndata: {JsonSerializer.Serialize(new { ok = true, symbol, res })}\n\n");
            await ctx.Response.Body.FlushAsync();

            var keepAlive = PeriodicTimerOrNull(TimeSpan.FromSeconds(10));

            try
            {
                while (!cancel.IsCancellationRequested)
                {
                    var readTask = ch.Reader.ReadAsync(cancel).AsTask();
                    if (keepAlive is null)
                    {
                        var payload = await readTask; // JSON already
                        await ctx.Response.WriteAsync($"data: {payload}\n\n");
                        await ctx.Response.Body.FlushAsync();
                        continue;
                    }

                    var kaTask = keepAlive.WaitForNextTickAsync(cancel).AsTask();
                    var completed = await Task.WhenAny(readTask, kaTask);
                    if (completed == readTask)
                    {
                        var payload = await readTask; // JSON already
                        await ctx.Response.WriteAsync($"data: {payload}\n\n");
                        await ctx.Response.Body.FlushAsync();
                    }
                    else
                    {
                        await ctx.Response.WriteAsync($": ping\n\n");
                        await ctx.Response.Body.FlushAsync();
                    }
                }
            }
            catch { /* client disconnected */ }
        });
    }

    private static PeriodicTimer? PeriodicTimerOrNull(TimeSpan? period)
        => period is null ? null : new PeriodicTimer(period.Value);
}

/// <summary>
/// Keeps a rolling bar store per (symbol,res), builds 1-minute bars from ticks,
/// fans out realtime events to SSE subscribers, and emits metrics periodically.
/// </summary>
public sealed class RealtimeHub : IHostedService, IDisposable
{
    private readonly ILogger _log;
    private readonly Func<MetricsSnapshot> _metrics; // provided by your bot
    private readonly ConcurrentDictionary<(string sym, string res), BarStore> _stores = new();
    private readonly ConcurrentDictionary<(string sym, string res), HashSet<ChannelWriter<string>>> _subs = new();
    private readonly JsonSerializerOptions _json = new(JsonSerializerDefaults.Web)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    private PeriodicTimer? _metricsTimer;
    private readonly CancellationTokenSource _cts = new();

    public RealtimeHub(ILogger<RealtimeHub> log, Func<MetricsSnapshot> metricsProvider)
    {
        _log = log;
        _metrics = metricsProvider;
    }

    public void Dispose() => _cts.Cancel();

    // ---------- PUBLIC API (called from your bot) ----------

    /// Seed historical bars (e.g., the 540 bars your DataFeed already built).
    public void SeedHistory(string symbol, string res, IEnumerable<Bar> bars)
        => Store(symbol, res).Seed(bars);

    /// Pass every tick (trade) here; we’ll update the current bar and broadcast.
    public void OnTick(string symbol, DateTime tsUtc, decimal price, long volume)
    {
        var store = Store(symbol, "1"); // start with 1m
        var changedBar = store.UpdateWithTick(tsUtc, price, volume);
        if (changedBar is null) return;

        var msg = JsonSerializer.Serialize(new
        {
            type = "bar",
            symbol,
            res = "1",
            bar = changedBar
        }, _json);

        Broadcast(symbol, "1", msg);
    }

    /// Update last price for positions to compute uPnL smoothly (optional).
    public void OnMark(string symbol, decimal lastPrice)
        => Store(symbol, "1").SetMark(lastPrice);

    // ---------- Query from /data/history ----------

    public IReadOnlyList<Bar> QueryHistory(string symbol, string res, long fromUnix, long toUnix)
        => Store(symbol, res).Query(fromUnix, toUnix);

    // ---------- SSE subscriptions ----------

    public (IDisposable sub, Channel<string> channel) Subscribe(string symbol, string res)
    {
        var channel = Channel.CreateUnbounded<string>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = false });

        var key = (symbol, res);
        var set = _subs.GetOrAdd(key, _ => new HashSet<ChannelWriter<string>>());

        lock (set) set.Add(channel.Writer);

        // Immediately push the last bar (if any) and a metrics snapshot
        var last = Store(symbol, res).LastBar();
        if (last is not null)
        {
            var msg = JsonSerializer.Serialize(new { type = "bar", symbol, res, bar = last }, _json);
            _ = channel.Writer.WriteAsync(msg);
        }

        var m = _metrics();
        var mm = JsonSerializer.Serialize(new { type = "metrics", data = m }, _json);
        _ = channel.Writer.WriteAsync(mm);

        var handle = new SubHandle(() =>
        {
            lock (set) set.Remove(channel.Writer);
            channel.Writer.TryComplete();
        });
        return (handle, channel);
    }

    // ---------- Hosted service: push metrics every second ----------

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _metricsTimer = new PeriodicTimer(TimeSpan.FromSeconds(1));
        _ = Task.Run(async () =>
        {
            try
            {
                while (await _metricsTimer!.WaitForNextTickAsync(_cts.Token))
                {
                    var m = _metrics();
                    var json = JsonSerializer.Serialize(new { type = "metrics", data = m }, _json);
                    BroadcastAll(json);
                }
            }
            catch { /* stopping */ }
        }, _cts.Token);

        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _cts.Cancel();
        _metricsTimer?.Dispose();
        return Task.CompletedTask;
    }

    // ---------- internals ----------

    private BarStore Store(string sym, string res)
        => _stores.GetOrAdd((sym, res), _ => new BarStore(res));

    private void Broadcast(string sym, string res, string json)
    {
        if (!_subs.TryGetValue((sym, res), out var set)) return;
        ChannelWriter<string>[] writers; lock (set) writers = set.ToArray();
        foreach (var w in writers) _ = w.WriteAsync(json);
    }

    private void BroadcastAll(string json)
    {
        foreach (var set in _subs.Values)
        {
            ChannelWriter<string>[] writers; lock (set) writers = set.ToArray();
            foreach (var w in writers) _ = w.WriteAsync(json);
        }
    }

    private sealed class SubHandle : IDisposable
    {
        private readonly Action _onDispose; public SubHandle(Action a) => _onDispose = a;
        public void Dispose() => _onDispose();
    }
}

// ======== Simple bar store / 1-minute builder ========

public sealed record Bar(
    long t,
    decimal o, decimal h, decimal l, decimal c,
    long v);

sealed class BarStore
{
    private readonly string _res; // e.g. "1" minutes
    private readonly LinkedList<Bar> _bars = new();
    private readonly object _sync = new();
    private decimal _mark;

    public BarStore(string res) => _res = res;

    public void Seed(IEnumerable<Bar> bars)
    {
        lock (_sync)
        {
            _bars.Clear();
            foreach (var b in bars.OrderBy(b => b.t))
                _bars.AddLast(b);
        }
    }

    public Bar? UpdateWithTick(DateTime tsUtc, decimal price, long vol)
    {
        var tOpen = AlignOpen(tsUtc);
        lock (_sync)
        {
            if (_bars.Last is null || _bars.Last.Value.t != tOpen)
            {
                var b = new Bar(tOpen, price, price, price, price, vol);
                _bars.AddLast(b);
                Trim(2000);
                return b;
            }
            else
            {
                var cur = _bars.Last.Value;
                var h = price > cur.h ? price : cur.h;
                var l = price < cur.l ? price : cur.l;
                var v = cur.v + vol;
                var b = cur with { h = h, l = l, c = price, v = v };
                _bars.Last.Value = b;
                return b;
            }
        }
    }

    public void SetMark(decimal last) { _mark = last; }

    public Bar? LastBar()
    {
        lock (_sync) return _bars.Last?.Value;
    }

    public IReadOnlyList<Bar> Query(long fromUnix, long toUnix)
    {
        lock (_sync)
        {
            return _bars.Where(b => b.t >= fromUnix && b.t <= toUnix).ToArray();
        }
    }

    private static long AlignOpen(DateTime tsUtc)
    {
        var t = tsUtc.ToUniversalTime();
        var open = new DateTime(t.Year, t.Month, t.Day, t.Hour, t.Minute, 0, DateTimeKind.Utc);
        return new DateTimeOffset(open).ToUnixTimeSeconds();
    }

    private void Trim(int max)
    {
        while (_bars.Count > max) _bars.RemoveFirst();
    }
}

// ======== Metrics snapshot pushed every second ========

public sealed record PositionChip(string sym, int qty, decimal avg, decimal mark, decimal uPnL, decimal rPnL);
public sealed record MetricsSnapshot(
    long accountId,
    string mode,
    decimal realized,
    decimal unrealized,
    decimal day,
    decimal maxDailyLoss,
    decimal remaining,
    string userHub,
    string marketHub,
    DateTime localTime,
    IReadOnlyList<PositionChip> positions);
