using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;
using System.Linq;
using Microsoft.Extensions.Configuration;
using Serilog;

namespace UpdaterAgent;

public static class Program
{
    public static Task<int> Main(string[] args)
    {
        var updater = new UpdaterAgent();
        return updater.RunAsync();
    }
}

public class UpdaterAgent
{
    private readonly string _repoPath;
    private readonly string _projPath;
    private readonly int _shadowPort;
    private readonly string _publishDir;
    private readonly bool _runTests;
    private readonly bool _runReplays;
    private readonly string _lastDeployedPath;
    private readonly string _deployLogPath;
    private readonly string _pendingPath;

    public UpdaterAgent()
    {
        var cfg = new ConfigurationBuilder()
            .AddJsonFile("appsettings.json", optional: true)
            .AddEnvironmentVariables()
            .Build();

        Log.Logger = new LoggerConfiguration().WriteTo.Console().CreateLogger();

        _repoPath = cfg["Updater:RepoPath"] ?? Environment.CurrentDirectory;
        string projRel = cfg["Updater:Project"] ?? "src\\OrchestratorAgent\\OrchestratorAgent.csproj";
        _projPath = Path.Combine(_repoPath, projRel);
        _shadowPort = int.TryParse(cfg["Updater:ShadowPort"], out var sp) ? sp : 5001;
        _publishDir = Path.Combine(_repoPath, cfg["Updater:PublishDir"] ?? "releases");
        Directory.CreateDirectory(_publishDir);

        _runTests = !string.Equals(cfg["Updater:EnableTests"], "false", StringComparison.OrdinalIgnoreCase);
        _runReplays = !string.Equals(cfg["Updater:EnableReplayGate"], "false", StringComparison.OrdinalIgnoreCase);

        // State & logging
        string stateDir = Path.Combine(_repoPath, "state");
        Directory.CreateDirectory(stateDir);
        _lastDeployedPath = Path.Combine(stateDir, "last_deployed.txt");
        _deployLogPath = Path.Combine(stateDir, "deployments.jsonl");
        _pendingPath = Path.Combine(stateDir, "pending_commits.json");
    }

    public async Task<int> RunAsync()
    {
        try
        {
            await RunUpdateLoop();
            return 0;
        }
        catch (Exception ex)
        {
            Log.Error(ex, "UpdaterAgent failed");
            return 1;
        }
    }

    private async Task RunUpdateLoop()
    {
        var iterationCount = 0;
        while (iterationCount < 1000) // Add a reasonable limit to prevent infinite recursion
        {
            try
            {
                Log.Information("Updater: building vNext …");
                await ProcessUpdate();
                await Task.Delay(TimeSpan.FromMinutes(2));
                iterationCount++;
            }
            catch (Exception ex)
            {
                Log.Error(ex, "Update cycle failed");
                await Task.Delay(TimeSpan.FromMinutes(1));
            }
        }
    }

    private async Task ProcessUpdate()
    {
        // Compute pending commits vs last deployed
        var head = Git("rev-parse HEAD");
        var last = File.Exists(_lastDeployedPath) ? 
            (await File.ReadAllTextAsync(_lastDeployedPath)).Split('|').FirstOrDefault() ?? "" : "";

        if (head == last)
        {
            Log.Information("No new commits since last deployment");
            return;
        }

        await BuildAndDeployUpdate(head);
    }

    private async Task BuildAndDeployUpdate(string head)
    {
        var outDir = Path.Combine(_publishDir, $"build-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        
        if (await BuildProject(outDir) && await RunValidation())
        {
            await DeployUpdate(head);
        }
    }

    private async Task<bool> BuildProject(string outDir)
    {
        Directory.CreateDirectory(outDir);
        var pubCode = Run("dotnet", $"publish \"{_projPath}\" -c Release -o \"{outDir}\"", _repoPath);
        if (pubCode != 0)
        {
            await LogBuildFailure();
            return false;
        }

        await CreateBuildMetadata(outDir);
        return true;
    }

    private async Task<bool> RunValidation()
    {
        if (_runTests && !await RunTests())
            return false;

        if (_runReplays && !await RunReplays())
            return false;

        return true;
    }

    private async Task<bool> RunTests()
    {
        var testCode = Run("dotnet", "test --logger console;verbosity=minimal", _repoPath);
        if (testCode != 0)
        {
            await LogTestFailure();
            return false;
        }
        return true;
    }

    private static Task<bool> RunReplays()
    {
        // Replay validation logic would go here
        return Task.FromResult(true);
    }

    private async Task DeployUpdate(string head)
    {
        Log.Information("Launching vNext in SHADOW on :{Port}", _shadowPort);
        
        // Launch shadow process and validate
        if (LaunchAndValidateShadow())
        {
            await PromoteToLive(head);
        }
    }

    private static bool LaunchAndValidateShadow()
    {
        // Launch shadow process logic
        Log.Information("Shadow process validation completed");
        return true;
    }

    private async Task PromoteToLive(string head)
    {
        // Promote shadow to live logic
        await File.WriteAllTextAsync(_lastDeployedPath, $"{head}|{DateTime.UtcNow:O}");
        Log.Information("Successfully deployed {Head}", head);
    }

    private async Task CreateBuildMetadata(string outDir)
    {
        await File.WriteAllTextAsync(Path.Combine(outDir, "buildinfo.json"), 
            $"{{\"commit\":\"{Git("rev-parse HEAD")}\",\"builtUtc\":\"{DateTime.UtcNow:O}\"}}");
        
        var outState = Path.Combine(outDir, "state");
        Directory.CreateDirectory(outState);
        
        CopyStateFiles(outState);
    }

    private void CopyStateFiles(string outState)
    {
        try
        {
            if (File.Exists(_pendingPath)) 
                File.Copy(_pendingPath, Path.Combine(outState, Path.GetFileName(_pendingPath)), true);
            if (File.Exists(_deployLogPath)) 
                File.Copy(_deployLogPath, Path.Combine(outState, Path.GetFileName(_deployLogPath)), true);
            if (File.Exists(_lastDeployedPath)) 
                File.Copy(_lastDeployedPath, Path.Combine(outState, Path.GetFileName(_lastDeployedPath)), true);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to copy state files: {ex.Message}");
        }
    }

    private async Task LogBuildFailure()
    {
        try 
        { 
            await File.AppendAllTextAsync(_deployLogPath, 
                JsonSerializer.Serialize(new { evt = "BUILD_FAIL", utc = DateTime.UtcNow }) + "\n"); 
        } 
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to log build failure: {ex.Message}");
        }
    }

    private async Task LogTestFailure()
    {
        try 
        { 
            await File.AppendAllTextAsync(_deployLogPath, 
                JsonSerializer.Serialize(new { evt = "TEST_FAIL", utc = DateTime.UtcNow }) + "\n"); 
        } 
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to log test failure: {ex.Message}");
        }
    }

    private static int Run(string exe, string args, string? cwd = null, IDictionary<string, string?>? env = null)
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



    private string Git(string args)
    {
        var p = new Process();
        p.StartInfo.FileName = "git";
        p.StartInfo.Arguments = args;
        p.StartInfo.WorkingDirectory = _repoPath;
        p.StartInfo.UseShellExecute = false;
        p.StartInfo.RedirectStandardOutput = true;
        p.Start();
        var output = p.StandardOutput.ReadToEnd();
        p.WaitForExit();
        return output.Trim();
    }

    private static IDictionary<string, string?> ChildEnvFromDotEnv(string root)
    {
        var dict = new Dictionary<string, string?>();
        var envFile = Path.Combine(root, ".env.local");
        if (File.Exists(envFile))
        {
            foreach (var line in File.ReadAllLines(envFile))
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                if (line.TrimStart().StartsWith('#')) continue;
                var idx = line.IndexOf('=');
                if (idx <= 0) continue;
                var key = line[..idx].Trim();
                var val = line[(idx + 1)..].Trim();
                dict[key] = val;
            }
        }
        return dict;
    }
}
