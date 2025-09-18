using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Config;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Infra;

public sealed class ParamStoreWatcher : IAsyncDisposable
{
    private readonly ILogger _log;
    private readonly FileSystemWatcher _fsw;
    private readonly TimeSpan _cooldown;
    private readonly bool _allowLive;
    private readonly ConcurrentDictionary<string, DateTime> _lastApply = new();
    private readonly CancellationTokenSource _cts = new();
    private bool _started;

    public ParamStoreWatcher(ILogger log, string paramsDir, TimeSpan cooldown, bool allowLive)
    {
        _log = log;
        _cooldown = cooldown <= TimeSpan.Zero ? TimeSpan.FromMinutes(5) : cooldown;
        _allowLive = allowLive;
        Directory.CreateDirectory(paramsDir);
        _fsw = new FileSystemWatcher(paramsDir, "S*-*.json")
        {
            NotifyFilter = NotifyFilters.LastWrite | NotifyFilters.CreationTime | NotifyFilters.FileName | NotifyFilters.Size,
            IncludeSubdirectories = false,
            EnableRaisingEvents = false
        };
        _fsw.Changed += OnFsEvent;
        _fsw.Created += OnFsEvent;
        _fsw.Renamed += OnFsEvent;
    }

    public void Start()
    {
        if (_started) return;
        _started = true;
        _fsw.EnableRaisingEvents = true;
        _log.LogInformation("[ParamWatch] Instant-apply watcher started (cooldown={Cooldown}s, allowLive={AllowLive})", (int)_cooldown.TotalSeconds, _allowLive);
    }

    private void OnFsEvent(object sender, FileSystemEventArgs e)
    {
        if (e.ChangeType == WatcherChangeTypes.Deleted) return;
        // Debounce per file
        _ = Task.Run(async () =>
        {
            try
            {
                await Task.Delay(500, _cts.Token).ConfigureAwait(false); // small debounce for atomic writes
                ApplyFromPathSafe(e.FullPath);
            }
            catch { }
        }, _cts.Token);
    }

    private void ApplyFromPathSafe(string path)
    {
        try
        {
            if (!_allowLive)
            {
                var live = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                if (live) { _log.LogInformation("[ParamWatch] Live mode detected — skipping instant apply for {File}", Path.GetFileName(path)); return; }
            }

            var name = Path.GetFileNameWithoutExtension(path);
            if (string.IsNullOrWhiteSpace(name)) return;
            var parts = name.Split('-', 2, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            if (parts.Length != 2) return;
            var strat = parts[0].ToUpperInvariant();
            var root = parts[1].ToUpperInvariant();

            var key = strat + ":" + root;
            var now = DateTime.UtcNow;
            var last = _lastApply.GetOrAdd(key, _ => DateTime.MinValue);
            if (now - last < _cooldown)
            {
                _log.LogDebug("[ParamWatch] Cooldown active for {Key} ({Remain}s)", key, (int)(_cooldown - (now - last)).TotalSeconds);
                return;
            }

            bool applied = strat switch
            {
                "S2" => ParamStore.ApplyS2OverrideIfPresent(root, _log),
                "S3" => ParamStore.ApplyS3OverrideIfPresent(root, _log),
                "S6" => ParamStore.ApplyS6OverrideIfPresent(root, _log),
                "S11" => ParamStore.ApplyS11OverrideIfPresent(root, _log),
                _ => false
            };
            if (applied)
            {
                _lastApply[key] = now;
                _log.LogInformation("[ParamWatch] Applied override instantly for {Key}", key);
            }
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "[ParamWatch] Failed to apply from path {Path}", path);
        }
    }

    public async ValueTask DisposeAsync()
    {
        try { _fsw.EnableRaisingEvents; } catch { }
        _fsw.Changed -= OnFsEvent;
        _fsw.Created -= OnFsEvent;
        _fsw.Renamed -= OnFsEvent;
        _fsw.Dispose();
        _cts.Cancel();
        await Task.CompletedTask.ConfigureAwait(false);
    }
}
