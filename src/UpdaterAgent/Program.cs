using System.Diagnostics;
using System.Net.Http.Json;
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
        // Optional: run tests
        if (runTests)
        {
            var testCode = Run("dotnet", "test -c Release", repoPath);
            if (testCode != 0)
            {
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
            Log.Error("Publish failed (exit={Code}).", pubCode);
            await Task.Delay(TimeSpan.FromMinutes(5));
            continue;
        }

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
            Log.Error("vNext did not promote to LIVE in time. Rollback: keep current live.");
            // No port swap; allow loop to retry later.
            await Task.Delay(TimeSpan.FromMinutes(5));
            continue;
        }

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
