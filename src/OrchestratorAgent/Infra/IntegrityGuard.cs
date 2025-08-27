#nullable enable
using System.Security.Cryptography;
using System.Text.Json;

namespace OrchestratorAgent.Infra
{
    /// <summary>
    /// Verifies integrity of critical auth/connection source files to prevent accidental changes from breaking login/JWT/SignalR.
    /// Behavior:
    /// - Looks for repo root by locating TopstepX.Bot.sln walking up from current directory.
    /// - Computes SHA256 over protected paths (files or directories of *.cs) and compares to state\\integrity.lock.json at repo root.
    /// - If lock missing: creates it with current hashes and logs notice.
    /// - If mismatch: in enforce mode (LIVE or INTEGRITY_ENFORCE=1), throws; otherwise logs warning and continues.
    /// Controls:
    /// - INTEGRITY_ENFORCE=1 to enforce in all modes; otherwise enforced automatically when LIVE.
    /// - INTEGRITY_RESET=1 to regenerate the lock file (use cautiously).
    /// </summary>
    internal static class IntegrityGuard
    {
        public static void EnsureLocked(IEnumerable<string> protectedPaths, bool liveMode, Microsoft.Extensions.Logging.ILogger log)
        {
            try
            {
                var enforce = (Environment.GetEnvironmentVariable("INTEGRITY_ENFORCE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes" || liveMode;
                var reset   = (Environment.GetEnvironmentVariable("INTEGRITY_RESET")   ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";

                var repo = FindRepoRoot();
                if (repo is null)
                {
                    log.LogWarning("[Integrity] Repo root not found. Skipping integrity check.");
                    return;
                }

                var stateDir = Path.Combine(repo, "state");
                Directory.CreateDirectory(stateDir);
                var lockPath = Path.Combine(stateDir, "integrity.lock.json");

                var current = ComputeHashes(repo, protectedPaths);

                if (reset || !File.Exists(lockPath))
                {
                    File.WriteAllText(lockPath, JsonSerializer.Serialize(current, new JsonSerializerOptions { WriteIndented = true }));
                    log.LogInformation("[Integrity] {Action} integrity lock at {Path} (protected {Count} entries).", reset?"Reset":"Created", lockPath, current.Count);
                    return;
                }

                var saved = JsonSerializer.Deserialize<Dictionary<string,string>>(File.ReadAllText(lockPath)) ?? new();

                var diffs = new List<string>();
                foreach (var kv in current)
                {
                    if (!saved.TryGetValue(kv.Key, out var hv) || !string.Equals(hv, kv.Value, StringComparison.OrdinalIgnoreCase))
                        diffs.Add($"{kv.Key} (current={kv.Value}, saved={(hv??"<missing>")})");
                }
                // also detect removed entries
                foreach (var kv in saved)
                {
                    if (!current.ContainsKey(kv.Key)) diffs.Add($"{kv.Key} (current=<missing>, saved={kv.Value})");
                }

                if (diffs.Count == 0)
                {
                    log.LogInformation("[Integrity] Protected auth/connection code verified (OK).");
                    return;
                }

                var msg = "[Integrity] Protected files changed:\n  - " + string.Join("\n  - ", diffs);
                if (enforce)
                {
                    log.LogError(msg + "\nSince enforcement is active (LIVE or INTEGRITY_ENFORCE=1), aborting launch to protect login/auth.");
                    throw new InvalidOperationException("Auth/connection integrity mismatch");
                }
                else
                {
                    log.LogWarning(msg + "\nContinuing (non-enforced mode). Set INTEGRITY_ENFORCE=1 or go LIVE to enforce.");
                }
            }
            catch (Exception ex)
            {
                log.LogWarning(ex, "[Integrity] Check failed (non-fatal).");
            }
        }

        private static string? FindRepoRoot()
        {
            try
            {
                var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
                for (var d = dir; d != null; d = d.Parent)
                {
                    if (File.Exists(Path.Combine(d.FullName, "TopstepX.Bot.sln"))) return d.FullName;
                }
            }
            catch { }
            return null;
        }

        private static Dictionary<string,string> ComputeHashes(string repoRoot, IEnumerable<string> protectedPaths)
        {
            var map = new Dictionary<string,string>(StringComparer.OrdinalIgnoreCase);
            foreach (var p in protectedPaths)
            {
                var full = Path.Combine(repoRoot, p);
                if (Directory.Exists(full))
                {
                    foreach (var file in Directory.EnumerateFiles(full, "*.cs", SearchOption.AllDirectories))
                    {
                        var rel = MakeRelative(repoRoot, file);
                        map[rel] = Sha256File(file);
                    }
                }
                else if (File.Exists(full))
                {
                    map[p] = Sha256File(full);
                }
            }
            return map;
        }

        private static string MakeRelative(string root, string path)
        {
            var rel = Path.GetRelativePath(root, path);
            return rel.Replace('/', '\\');
        }

        private static string Sha256File(string file)
        {
            using var sha = SHA256.Create();
            using var fs = File.OpenRead(file);
            var hash = sha.ComputeHash(fs);
            return Convert.ToHexString(hash);
        }
    }
}
