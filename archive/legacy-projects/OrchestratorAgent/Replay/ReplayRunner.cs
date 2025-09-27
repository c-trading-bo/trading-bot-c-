using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

internal sealed class ReplayRunner(Action<ReplayRunner.TradeTick> onTick)
{
    private readonly Action<TradeTick> _onTick = onTick;

    public async Task RunAsync(string file, TimeSpan? maxDuration, CancellationToken ct)
    {
        if (!File.Exists(file)) return;
        var text = await File.ReadAllTextAsync(file, ct).ConfigureAwait(false);
        var ticks = JsonSerializer.Deserialize<List<TradeTick>>(text) ?? [];
        var start = DateTime.UtcNow;
        foreach (var t in ticks)
        {
            _onTick(t);
            if (maxDuration.HasValue && DateTime.UtcNow - start > maxDuration.Value) break;
            await Task.Yield().ConfigureAwait(false);
        }
    }

    internal record TradeTick(string Symbol, DateTime ExchangeTimeUtc, decimal Price, int Size);
}
