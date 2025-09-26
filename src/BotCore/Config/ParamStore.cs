using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace BotCore.Config;

public static class ParamStore
{
    private const string EvtSave = "save";
    private const string EvtApply = "apply";
    public sealed record S2Override(string SymbolRoot, DateTime ExpiresUtc, Dictionary<string, JsonElement> Extra);
    public sealed record S3Override(string SymbolRoot, DateTime ExpiresUtc, string JsonConfig);
    public sealed record S6Override(string SymbolRoot, DateTime ExpiresUtc, Dictionary<string, JsonElement> Extra);
    public sealed record S11Override(string SymbolRoot, DateTime ExpiresUtc, Dictionary<string, JsonElement> Extra);

    private static string ParamsDir()
        => Path.Combine(AppContext.BaseDirectory, "state", "params");

    private static string S2Path(string root)
        => Path.Combine(ParamsDir(), $"S2-{root.ToUpperInvariant()}.json");
    private static string S3Path(string root)
        => Path.Combine(ParamsDir(), $"S3-{root.ToUpperInvariant()}.json");
    private static string S6Path(string root)
        => Path.Combine(ParamsDir(), $"S6-{root.ToUpperInvariant()}.json");
    private static string S11Path(string root)
        => Path.Combine(ParamsDir(), $"S11-{root.ToUpperInvariant()}.json");

    // In-memory de-dupe of last-applied overrides to prevent duplicate apply/logs
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, string> _lastAppliedHash = new();

    private static bool ShouldApply(string strat, string root, string payloadJson, DateTime expiresUtc)
    {
        try
        {
            var key = $"{strat}:{root.ToUpperInvariant()}";
            var stamp = $"{expiresUtc.ToUniversalTime():u}|{payloadJson}";
            // Atomically set and compare previous value to avoid race double-applies
            var prev = _lastAppliedHash.AddOrUpdate(key, stamp, (_, old) => stamp);
            if (string.Equals(prev, stamp, StringComparison.Ordinal))
                return false; // nothing changed, skip
            return true;
        }
        catch { return true; }
    }

    private static void AppendHistory(string evt, string strat, string root, DateTime? expiresUtc, object payload)
    {
        try
        {
            var dir = ParamsDir();
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, "history.jsonl");
            var rec = new
            {
                ts = DateTime.UtcNow,
                evt,
                strat,
                root = root.ToUpperInvariant(),
                expiresUtc,
                payload
            };
            var line = JsonSerializer.Serialize(rec) + Environment.NewLine;
            File.AppendAllText(path, line);
        }
        catch (Exception)
        {
            // Best-effort logging only; ignore IO issues
        }
    }

    public static void SaveS2(string root, Dictionary<string, JsonElement> extra, TimeSpan ttl)
    {
        ArgumentNullException.ThrowIfNull(root);
        ArgumentNullException.ThrowIfNull(extra);
        
        try
        {
            Directory.CreateDirectory(ParamsDir());
            var exp = DateTime.UtcNow.Add(ttl);
            var payload = new S2Override(root.ToUpperInvariant(), exp, extra);
            var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(S2Path(root), json);
            AppendHistory(EvtSave, "S2", root, exp, new { keys = extra.Keys });
        }
        catch (Exception)
        {
            // Best-effort save; ignore IO issues
        }
    }

    public static bool TryLoadS2(string root, out S2Override ov)
    {
        ArgumentNullException.ThrowIfNull(root);
        
        ov = default!;
        try
        {
            var path = S2Path(root);
            if (!File.Exists(path)) return false;
            var json = File.ReadAllText(path);
            var obj = JsonSerializer.Deserialize<S2Override>(json);
            if (obj is null) return false;
            if (obj.ExpiresUtc <= DateTime.UtcNow) return false;
            ov = obj;
            return true;
        }
        catch { return false; }
    }

    public static bool ApplyS2OverrideIfPresent(string root, Microsoft.Extensions.Logging.ILogger? log = null)
    {
        try
        {
            if (!TryLoadS2(root, out var ov)) return false;
            // Dedupe: skip if identical payload already applied for same expiry
            var payloadJson = JsonSerializer.Serialize(ov.Extra);
            if (!ShouldApply("S2", root, payloadJson, ov.ExpiresUtc)) return false;
            var def = new StrategyDef { Id = "S2", Name = "S2-Override", Enabled = true };
            foreach (var kv in ov.Extra) def.Extra[kv.Key] = kv.Value;
            Strategy.S2RuntimeConfig.ApplyFrom(def);
            log?.LogInformation("[ParamStore] Applied S2 override for {Root} (expires {Exp:u})", root, ov.ExpiresUtc);
            AppendHistory(EvtApply, "S2", root, ov.ExpiresUtc, new { keys = ov.Extra.Keys });
            return true;
        }
        catch
        {
            return false;
        }
    }

    public static void SaveS3(string root, string jsonConfig, TimeSpan ttl)
    {
        try
        {
            Directory.CreateDirectory(ParamsDir());
            var exp = DateTime.UtcNow.Add(ttl);
            var payload = new S3Override(root.ToUpperInvariant(), exp, jsonConfig);
            var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(S3Path(root), json);
            AppendHistory(EvtSave, "S3", root, exp, new { length = jsonConfig?.Length ?? 0 });
        }
        catch (System.Exception)
        {
            // Best-effort save; ignore IO issues
        }
    }

    public static bool ApplyS3OverrideIfPresent(string root, Microsoft.Extensions.Logging.ILogger? log = null)
    {
        try
        {
            var path = S3Path(root);
            if (!File.Exists(path)) return false;
            var json = File.ReadAllText(path);
            var obj = JsonSerializer.Deserialize<S3Override>(json);
            if (obj is null || obj.ExpiresUtc <= DateTime.UtcNow) return false;
            if (string.IsNullOrWhiteSpace(obj.JsonConfig)) return false;
            if (!ShouldApply("S3", root, obj.JsonConfig, obj.ExpiresUtc)) return false;
            Strategy.S3Strategy.ApplyTuningJson(obj.JsonConfig);
            log?.LogInformation("[ParamStore] Applied S3 override for {Root} (expires {Exp:u})", root, obj.ExpiresUtc);
            AppendHistory(EvtApply, "S3", root, obj.ExpiresUtc, new { length = obj.JsonConfig?.Length ?? 0 });
            return true;
        }
        catch (System.Exception)
        {
            return false;
        }
    }

    public static void SaveS6(string root, Dictionary<string, JsonElement> extra, TimeSpan ttl)
    {
        try
        {
            Directory.CreateDirectory(ParamsDir());
            var exp = DateTime.UtcNow.Add(ttl);
            var payload = new S6Override(root.ToUpperInvariant(), exp, extra);
            var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(S6Path(root), json);
            AppendHistory(EvtSave, "S6", root, exp, new { keys = extra.Keys });
        }
        catch (System.Exception)
        {
            // Best-effort save; ignore IO issues
        }
    }

    public static bool ApplyS6OverrideIfPresent(string root, Microsoft.Extensions.Logging.ILogger? log = null)
    {
        try
        {
            var path = S6Path(root);
            if (!File.Exists(path)) return false;
            var json = File.ReadAllText(path);
            var obj = JsonSerializer.Deserialize<S6Override>(json);
            if (obj is null || obj.ExpiresUtc <= DateTime.UtcNow) return false;
            var payloadJson = JsonSerializer.Serialize(obj.Extra);
            if (!ShouldApply("S6", root, payloadJson, obj.ExpiresUtc)) return false;
            var def = new StrategyDef { Id = "S6", Name = "S6-Override", Enabled = true };
            foreach (var kv in obj.Extra) def.Extra[kv.Key] = kv.Value;
            Strategy.S6RuntimeConfig.ApplyFrom(def);
            log?.LogInformation("[ParamStore] Applied S6 override for {Root} (expires {Exp:u})", root, obj.ExpiresUtc);
            AppendHistory(EvtApply, "S6", root, obj.ExpiresUtc, new { keys = obj.Extra.Keys });
            return true;
        }
        catch (System.Exception)
        {
            return false;
        }
    }

    public static void SaveS11(string root, Dictionary<string, JsonElement> extra, TimeSpan ttl)
    {
        try
        {
            Directory.CreateDirectory(ParamsDir());
            var exp = DateTime.UtcNow.Add(ttl);
            var payload = new S11Override(root.ToUpperInvariant(), exp, extra);
            var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(S11Path(root), json);
            AppendHistory(EvtSave, "S11", root, exp, new { keys = extra.Keys });
        }
        catch (System.Exception)
        {
            // Best-effort save; ignore IO issues
        }
    }

    public static bool ApplyS11OverrideIfPresent(string root, Microsoft.Extensions.Logging.ILogger? log = null)
    {
        try
        {
            var path = S11Path(root);
            if (!File.Exists(path)) return false;
            var json = File.ReadAllText(path);
            var obj = JsonSerializer.Deserialize<S11Override>(json);
            if (obj is null || obj.ExpiresUtc <= DateTime.UtcNow) return false;
            var payloadJson = JsonSerializer.Serialize(obj.Extra);
            if (!ShouldApply("S11", root, payloadJson, obj.ExpiresUtc)) return false;
            var def = new StrategyDef { Id = "S11", Name = "S11-Override", Enabled = true };
            foreach (var kv in obj.Extra) def.Extra[kv.Key] = kv.Value;
            Strategy.S11RuntimeConfig.ApplyFrom(def);
            log?.LogInformation("[ParamStore] Applied S11 override for {Root} (expires {Exp:u})", root, obj.ExpiresUtc);
            AppendHistory(EvtApply, "S11", root, obj.ExpiresUtc, new { keys = obj.Extra.Keys });
            return true;
        }
        catch (System.Exception)
        {
            return false;
        }
    }
}
