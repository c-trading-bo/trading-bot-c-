using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

public sealed class ReplayRunner
{
    private readonly Action<TradeTick> _onTick;
    public ReplayRunner(Action<TradeTick> onTick) { _onTick = onTick; }

    public async Task RunAsync(string file, TimeSpan? maxDuration, CancellationToken ct)
    {
        if (!File.Exists(file)) return;
        var text = await File.ReadAllTextAsync(file, ct);
        var ticks = JsonSerializer.Deserialize<List<TradeTick>>(text) ?? new();
        var start = DateTime.UtcNow;
        foreach (var t in ticks)
        {
            _onTick(t);
            if (maxDuration.HasValue && DateTime.UtcNow - start > maxDuration.Value) break;
            await Task.Yield();
        }
    }

    public record TradeTick(string Symbol, DateTime ExchangeTimeUtc, decimal Price, int Size);
}
