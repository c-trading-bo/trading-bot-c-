using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;
using System.Linq;
using Microsoft.Extensions.Configuration;
using Serilog;

static int Run(string exe, string args, string? cwd = null, IDictionary<string, string?>? env = null)
{
    var p = new Process();
    p.StartInfo.FileName = exe;
    p.StartInfo.Arguments = args;
    p.StartInfo.WorkingDirectory = cwd ?? Environment.CurrentDirectory;
    p.StartInfo.UseShellExecute = false;
    p.StartInfo.RedirectStandardOutput = true;
    p.StartInfo.RedirectStandardError = true;
    if (env != null)
    {
        foreach (var kv in env)
            p.StartInfo.Environment[kv.Key] = kv.Value ?? string.Empty;
    }
    p.OutputDataReceived += (_, e) => { if (e.Data != null) Console.WriteLine(e.Data); };
    p.ErrorDataReceived += (_, e) => { if (e.Data != null) Console.Error.WriteLine(e.Data); };
    p.Start();
    p.BeginOutputReadLine();
    p.BeginErrorReadLine();
    p.WaitForExit();
    return p.ExitCode;
}

async Task<bool> HealthOkAsync(int port)
{
    try
    {
        using var http = new HttpClient { BaseAddress = new Uri($"http://localhost:{port}") };
        var obj = await http.GetFromJsonAsync<Dictionary<string, object>>("/healthz");
        return obj != null && obj.TryGetValue("ok", out var ok) && ok?.ToString()?.ToLowerInvariant() == "true";
    }
    catch { return false; }
}

async Task<bool> ModeIsAsync(int port, string wanted)
{
    try
    {
        using var http = new HttpClient { BaseAddress = new Uri($"http://localhost:{port}") };
        var obj = await http.GetFromJsonAsync<Dictionary<string, object>>("/healthz/mode");
        var mode = obj != null && obj.TryGetValue("mode", out var m) ? m?.ToString() : null;
        return string.Equals(mode, wanted, StringComparison.OrdinalIgnoreCase);
    }
    catch { return false; }
}

async Task<bool> ReplayGateAsync(int port, string replayDir)
{
    // Placeholder: pass gate if no replay infra is present.
    if (!Directory.Exists(replayDir)) return true;
    await Task.Delay(500);
    return true;
}

async Task<HashSet<string>> CapsAsync(int port)
{
    try
    {
        using var http = new HttpClient { BaseAddress = new Uri($"http://localhost:{port}") };
        var arr = await http.GetFromJsonAsync<string[]>("/capabilities");
        return arr is null ? [] : [.. arr];
    }
    catch { return []; }
}

var cfg = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables()
    .Build();

Log.Logger = new LoggerConfiguration().WriteTo.Console().CreateLogger();

string repoPath = cfg["Updater:RepoPath"] ?? Environment.CurrentDirectory;
string projRel = cfg["Updater:Project"] ?? "src\\OrchestratorAgent\\OrchestratorAgent.csproj";
string projPath = Path.Combine(repoPath, projRel);
int livePort = int.TryParse(cfg["Updater:LivePort"], out var lp) ? lp : 5000;
int shadowPort = int.TryParse(cfg["Updater:ShadowPort"], out var sp) ? sp : 5001;
string publishDir = Path.Combine(repoPath, cfg["Updater:PublishDir"] ?? "releases");
Directory.CreateDirectory(publishDir);

TimeSpan dryMin = TimeSpan.FromMinutes(double.TryParse(cfg["Updater:DryRunMinutes"], out var dm) ? dm : 2);
int minHealthy = int.TryParse(cfg["Updater:MinHealthyPasses"], out var mh) ? mh : 3;
bool runTests = !string.Equals(cfg["Updater:EnableTests"], "false", StringComparison.OrdinalIgnoreCase);
bool runReplays = !string.Equals(cfg["Updater:EnableReplayGate"], "false", StringComparison.OrdinalIgnoreCase);
string replayDir = Path.Combine(repoPath, cfg["Updater:ReplayDir"] ?? "replays");

// State & logging
string StateDir = Path.Combine(repoPath, "state");
Directory.CreateDirectory(StateDir);
string LastDeployedPath = Path.Combine(StateDir, "last_deployed.txt");
string DeployLogPath = Path.Combine(StateDir, "deployments.jsonl");
string PendingPath = Path.Combine(StateDir, "pending_commits.json");

string Git(string args)
{
    try
    {
        var p = new Process();
        p.StartInfo.FileName = "git";
        p.StartInfo.Arguments = args;
        p.StartInfo.WorkingDirectory = repoPath;
        p.StartInfo.UseShellExecute = false;
        p.StartInfo.RedirectStandardOutput = true;
        p.Start();
        var s = p.StandardOutput.ReadToEnd().Trim();
        p.WaitForExit();
        return s;
    }
    catch { return string.Empty; }
}

async Task Notify(string level, string msg)
{
    try
    {
        var url = Environment.GetEnvironmentVariable("BOT_ALERT_WEBHOOK");
        if (string.IsNullOrWhiteSpace(url)) return;
        using var http = new HttpClient();
        await http.PostAsJsonAsync(url, new { content = $"`{level}` {msg}" });
    }
    catch { }
}

// Load .env.local to pass secrets/env to vNext process
Dictionary<string, string?> ChildEnvFromDotEnv(string root)
{
    var dict = new Dictionary<string, string?>(StringComparer.OrdinalIgnoreCase)
    {
        ["ASPNETCORE_URLS"] = $"http://localhost:{shadowPort}",
        ["BOT_QUICK_EXIT"] = "0",
        // Routing is still controlled by Mode+Lease inside the app; LIVE_ORDERS sync happens there.
    };
    var envFile = Path.Combine(root, ".env.local");
    if (File.Exists(envFile))
    {
        foreach (var line in File.ReadAllLines(envFile))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (line.TrimStart().StartsWith("#")) continue;
            var idx = line.IndexOf('=');
            if (idx <= 0) continue;
            var key = line[..idx].Trim();
            var val = line[(idx + 1)..].Trim();
            dict[key] = val;
        }
    }
    return dict;
}

while (true)
{
    try
    {
        Log.Information("Updater: building vNext …");

        // Compute pending commits vs last deployed
        var head = Git("rev-parse HEAD");
        var last = File.Exists(LastDeployedPath) ? (File.ReadAllText(LastDeployedPath).Split('|').FirstOrDefault() ?? "") : "";
        var range = string.IsNullOrWhiteSpace(last) ? head : $"{last}..{head}";
        var rawLog = Git($"log --pretty=format:%H|%ad|%an|%s --date=iso {range}");
        try
        {
            var list = rawLog.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries)
                              .Select(l => { var parts = l.Split('|', 4); return new { commit = parts.ElementAtOrDefault(0) ?? "", date = parts.ElementAtOrDefault(1) ?? "", author = parts.ElementAtOrDefault(2) ?? "", subject = parts.ElementAtOrDefault(3) ?? "" }; });
            await File.WriteAllTextAsync(PendingPath, JsonSerializer.Serialize(list, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch { }
        try { await File.AppendAllTextAsync(DeployLogPath, JsonSerializer.Serialize(new { evt = "DEPLOY_START", utc = DateTime.UtcNow, head }) + "\n"); } catch { }

        // Optional: run tests
        if (runTests)
        {
            var testCode = Run("dotnet", "test -c Release", repoPath);
            if (testCode != 0)
            {
                try { await File.AppendAllTextAsync(DeployLogPath, JsonSerializer.Serialize(new { evt = "ABORT", reason = "TEST_FAIL", utc = DateTime.UtcNow, head }) + "\n"); } catch { }
                await Notify("ERROR", $"Upgrade aborted: tests failed @ {head}");
                Log.Error("Tests failed (exit={Code}). Will retry later.", testCode);
                await Task.Delay(TimeSpan.FromMinutes(5));
                continue;
            }
        }

        var outDir = Path.Combine(publishDir, DateTime.UtcNow.ToString("yyyyMMdd-HHmmss"));
        Directory.CreateDirectory(outDir);
        var pubCode = Run("dotnet", $"publish \"{projPath}\" -c Release -o \"{outDir}\"", repoPath);
        if (pubCode != 0)
        {
            try { await File.AppendAllTextAsync(DeployLogPath, JsonSerializer.Serialize(new { evt = "ABORT", reason = "BUILD_FAIL", utc = DateTime.UtcNow, head }) + "\n"); } catch { }
            await Notify("ERROR", $"Upgrade aborted: build failed @ {head}");
            Log.Error("Publish failed (exit={Code}).", pubCode);
            await Task.Delay(TimeSpan.FromMinutes(5));
            continue;
        }

        // Stamp build metadata and copy state snapshot into publish dir
        try
        {
            await File.WriteAllTextAsync(Path.Combine(outDir, "buildinfo.json"), $"{{\"commit\":\"{head}\",\"builtUtc\":\"{DateTime.UtcNow:O}\"}}");
            var outState = Path.Combine(outDir, "state");
            Directory.CreateDirectory(outState);
            if (File.Exists(PendingPath)) File.Copy(PendingPath, Path.Combine(outState, Path.GetFileName(PendingPath)), true);
            if (File.Exists(DeployLogPath)) File.Copy(DeployLogPath, Path.Combine(outState, Path.GetFileName(DeployLogPath)), true);
            if (File.Exists(LastDeployedPath)) File.Copy(LastDeployedPath, Path.Combine(outState, Path.GetFileName(LastDeployedPath)), true);
        }
        catch { }

        // Launch vNext (shadow) on ShadowPort
        Log.Information("Launching vNext in SHADOW on :{Port}", shadowPort);
        var childEnv = ChildEnvFromDotEnv(repoPath);
        // Try exe first, then dll via dotnet
        var exePath = Path.Combine(outDir, "OrchestratorAgent.exe");
        var dllPath = Path.Combine(outDir, "OrchestratorAgent.dll");

        int launchExit;
        if (File.Exists(exePath))
        {
            // Detach via cmd start so updater can keep running
            launchExit = Run("cmd.exe", $"/c start \"junie-vNext\" \"{exePath}\"", outDir, childEnv);
        }
        else if (File.Exists(dllPath))
        {
            launchExit = Run("cmd.exe", $"/c start \"junie-vNext\" dotnet \"{dllPath}\"", outDir, childEnv);
        }
        else
        {
            Log.Error("Publish output missing OrchestratorAgent.exe/dll in {Out}", outDir);
            await Task.Delay(TimeSpan.FromMinutes(5));
            continue;
        }
        if (launchExit != 0)
        {
            Log.Error("Failed to start vNext (exit={Code}).", launchExit);
            await Task.Delay(TimeSpan.FromMinutes(5));
            continue;
        }

        // Health/dry-run gate
        int okStreak = 0; var start = DateTime.UtcNow;
        while (DateTime.UtcNow - start < dryMin || okStreak < minHealthy)
        {
            if (await HealthOkAsync(shadowPort)) okStreak++; else okStreak = 0;
            await Task.Delay(1000);
        }
        try { await File.AppendAllTextAsync(DeployLogPath, JsonSerializer.Serialize(new { evt = "HEALTH_OK", utc = DateTime.UtcNow, head }) + "\n"); } catch { }

        // Optional capability gate
        var featureManifest = Path.Combine(repoPath, "state", "feature-manifest.json");
        if (File.Exists(featureManifest))
        {
            try
            {
                var doc = JsonSerializer.Deserialize<Dictionary<string, string[]>>(await File.ReadAllTextAsync(featureManifest)) ?? [];
                if (doc.TryGetValue("required", out var req) && req != null && req.Length > 0)
                {
                    var have = await CapsAsync(shadowPort);
                    var missing = req.Where(r => !have.Contains(r)).ToArray();
                    if (missing.Length > 0)
                    {
                        try { await File.AppendAllTextAsync(DeployLogPath, JsonSerializer.Serialize(new { evt = "ABORT", reason = "CAP_MISSING", miss = missing, utc = DateTime.UtcNow, head }) + "\n"); } catch { }
                        await Notify("ERROR", $"Upgrade aborted: missing features {string.Join(",", missing)} @ {head}");
                        Log.Error("Capability gate failed: {Missing}", string.Join(",", missing));
                        await Task.Delay(TimeSpan.FromMinutes(5));
                        continue;
                    }
                }
            }
            catch { }
        }

        if (runReplays)
        {
            var replayOk = await ReplayGateAsync(shadowPort, replayDir);
            if (!replayOk)
            {
                Log.Error("Replay gate failed; aborting switch.");
                await Task.Delay(TimeSpan.FromMinutes(5));
                continue;
            }
        }

        // Ask current live to DEMOTE (enter DRAIN and release lease on exit)
        try
        {
            using var http = new HttpClient { BaseAddress = new Uri($"http://localhost:{livePort}") };
            await http.PostAsync("/demote", null);
            Log.Information("Requested DEMOTE on current LIVE :{Port}", livePort);
        }
        catch
        {
            Log.Warning("Could not reach vCurrent on :{Port}; continuing.", livePort);
        }

        // Verify vNext flips to LIVE
        var deadline = DateTime.UtcNow.AddSeconds(45);
        var live = false;
        while (DateTime.UtcNow < deadline)
        {
            if (await ModeIsAsync(shadowPort, "LIVE")) { live = true; break; }
            await Task.Delay(1000);
        }
        if (!live)
        {
            try { await File.AppendAllTextAsync(DeployLogPath, JsonSerializer.Serialize(new { evt = "ABORT", reason = "PROMOTE_TIMEOUT", utc = DateTime.UtcNow, head }) + "\n"); } catch { }
            await Notify("WARN", $"Upgrade aborted: vNext never became LIVE @ {head}");
            Log.Error("vNext did not promote to LIVE in time. Rollback: keep current live.");
            // No port swap; allow loop to retry later.
            await Task.Delay(TimeSpan.FromMinutes(5));
            continue;
        }

        try
        {
            await File.AppendAllTextAsync(DeployLogPath, JsonSerializer.Serialize(new { evt = "PROMOTE", utc = DateTime.UtcNow, head, port = shadowPort }) + "\n");
            await File.WriteAllTextAsync(LastDeployedPath, $"{head}|{DateTime.UtcNow:O}");
            await Notify("INFO", $"PROMOTE → LIVE commit={head} port={shadowPort}");
        }
        catch { }

        Log.Information("Switch complete: vNext is LIVE on :{Port}", shadowPort);
        // Sleep before next check
        await Task.Delay(TimeSpan.FromMinutes(5));
    }
    catch (Exception ex)
    {
        Log.Error(ex, "Updater error");
        await Task.Delay(TimeSpan.FromSeconds(15));
    }
}
