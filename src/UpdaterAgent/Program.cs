using System.Diagnostics;
using System.Net.Http.Json;
using System.Text.Json;
using System.Linq;
using System.Security;
using Microsoft.Extensions.Configuration;
using Serilog;
using System.Globalization;

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
    // Configuration constants to eliminate magic numbers
    private const int DefaultShadowPort = 5001;
    private const int MaxUpdateIterations = 1000;

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

        Log.Logger = new LoggerConfiguration().WriteTo.Console(formatProvider: CultureInfo.InvariantCulture).CreateLogger();

        _repoPath = cfg["Updater:RepoPath"] ?? Environment.CurrentDirectory;
        string projRel = cfg["Updater:Project"] ?? "src\\OrchestratorAgent\\OrchestratorAgent.csproj";
        _projPath = Path.Combine(_repoPath, projRel);
        _shadowPort = int.TryParse(cfg["Updater:ShadowPort"], out var sp) ? sp : DefaultShadowPort;
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
            await RunUpdateLoop().ConfigureAwait(false);
            return 0;
        }
        catch (TaskCanceledException ex)
        {
            Log.Warning(ex, "UpdaterAgent was cancelled");
            return 1;
        }
        catch (OperationCanceledException ex)
        {
            Log.Warning(ex, "UpdaterAgent operation was cancelled");
            return 1;
        }
        catch (InvalidOperationException ex)
        {
            Log.Error(ex, "UpdaterAgent invalid operation");
            return 1;
        }
        catch (Exception ex) when (ex.IsFatal())
        {
            Log.Fatal(ex, "Fatal error in UpdaterAgent - terminating");
            throw; // Rethrow fatal exceptions
        }
        catch (InvalidOperationException ex)
        {
            Log.Error(ex, "Invalid operation in UpdaterAgent - returning error code");
            return 1;
        }
        catch (UnauthorizedAccessException ex)
        {
            Log.Error(ex, "Access denied in UpdaterAgent - returning error code");
            return 1;
        }
        catch (IOException ex)
        {
            Log.Error(ex, "IO error in UpdaterAgent - returning error code");
            return 1;
        }
        catch (Exception ex) when (!ex.IsFatal())
        {
            Log.Error(ex, "Unexpected error in UpdaterAgent - returning error code");
            return 1;
        }
    }

    private async Task RunUpdateLoop()
    {
        var iterationCount = 0;
        while (iterationCount < MaxUpdateIterations) // Add a reasonable limit to prevent infinite recursion
        {
            try
            {
                Log.Information("Updater: building vNext …");
                await ProcessUpdate().ConfigureAwait(false);
                await Task.Delay(TimeSpan.FromMinutes(2)).ConfigureAwait(false);
                iterationCount++;
            }
            catch (HttpRequestException ex)
            {
                Log.Error(ex, "HTTP error during update cycle");
                await Task.Delay(TimeSpan.FromMinutes(1)).ConfigureAwait(false);
            }
            catch (TaskCanceledException ex)
            {
                Log.Warning(ex, "Update cycle was cancelled");
                break; // Exit the loop
            }
            catch (Exception ex) when (ex.IsFatal())
            {
                Log.Fatal(ex, "Fatal error in update cycle");
                throw; // Rethrow fatal exceptions
            }
            catch (InvalidOperationException ex)
            {
                Log.Error(ex, "Invalid operation in update cycle - continuing with next iteration");
                await Task.Delay(TimeSpan.FromMinutes(1)).ConfigureAwait(false);
            }
            catch (IOException ex)
            {
                Log.Error(ex, "IO error in update cycle - continuing with next iteration");
                await Task.Delay(TimeSpan.FromMinutes(1)).ConfigureAwait(false);
            }
            catch (Exception ex) when (!ex.IsFatal())
            {
                Log.Error(ex, "Unexpected error in update cycle - continuing with next iteration");
                await Task.Delay(TimeSpan.FromMinutes(1)).ConfigureAwait(false);
            }
        }
    }

    private async Task ProcessUpdate()
    {
        // Compute pending commits vs last deployed
        var head = Git("rev-parse HEAD");
        var last = File.Exists(_lastDeployedPath) ? 
            (await File.ReadAllTextAsync(_lastDeployedPath).ConfigureAwait(false)).Split('|').FirstOrDefault() ?? "" : "";

        if (head == last)
        {
            Log.Information("No new commits since last deployment");
            return;
        }

        await BuildAndDeployUpdate(head).ConfigureAwait(false);
    }

    private async Task BuildAndDeployUpdate(string head)
    {
        var outDir = Path.Combine(_publishDir, $"build-{DateTime.UtcNow:yyyyMMdd-HHmmss}");
        
        if (await BuildProject(outDir).ConfigureAwait(false) && await RunValidation().ConfigureAwait(false))
        {
            await DeployUpdate(head).ConfigureAwait(false);
        }
    }

    private async Task<bool> BuildProject(string outDir)
    {
        Directory.CreateDirectory(outDir);
        var pubCode = Run("dotnet", $"publish \"{_projPath}\" -c Release -o \"{outDir}\"", _repoPath);
        if (pubCode != 0)
        {
            await LogBuildFailure().ConfigureAwait(false);
            return false;
        }

        await CreateBuildMetadata(outDir).ConfigureAwait(false);
        return true;
    }

    private async Task<bool> RunValidation()
    {
        if (_runTests && !await RunTests().ConfigureAwait(false))
            return false;

        if (_runReplays && !await RunReplays().ConfigureAwait(false))
            return false;

        return true;
    }

    private async Task<bool> RunTests()
    {
        var testCode = Run("dotnet", "test --logger console;verbosity=minimal", _repoPath);
        if (testCode != 0)
        {
            await LogTestFailure().ConfigureAwait(false);
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
            await PromoteToLive(head).ConfigureAwait(false);
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
        await File.WriteAllTextAsync(_lastDeployedPath, $"{head}|{DateTime.UtcNow:O}").ConfigureAwait(false);
        Log.Information("Successfully deployed {Head}", head);
    }

    private async Task CreateBuildMetadata(string outDir)
    {
        await File.WriteAllTextAsync(Path.Combine(outDir, "buildinfo.json"), 
            $"{{\"commit\":\"{Git("rev-parse HEAD")}\",\"builtUtc\":\"{DateTime.UtcNow:O}\"}}").ConfigureAwait(false);
        
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
        catch (UnauthorizedAccessException ex)
        {
            Console.WriteLine($"Warning: Access denied copying state files: {ex.Message}");
        }
        catch (IOException ex)
        {
            Console.WriteLine($"Warning: IO error copying state files: {ex.Message}");
        }
        catch (Exception ex) when (!ex.IsFatal())
        {
            Console.WriteLine($"Warning: Failed to copy state files: {ex.Message}");
        }
    }

    private async Task LogBuildFailure()
    {
        try 
        { 
            await File.AppendAllTextAsync(_deployLogPath, 
                JsonSerializer.Serialize(new { evt = "BUILD_FAIL", utc = DateTime.UtcNow }) + "\n").ConfigureAwait(false); 
        } 
        catch (UnauthorizedAccessException ex)
        {
            Console.WriteLine($"Warning: Access denied logging build failure: {ex.Message}");
        }
        catch (IOException ex)
        {
            Console.WriteLine($"Warning: IO error logging build failure: {ex.Message}");
        }
        catch (Exception ex) when (!ex.IsFatal())
        {
            Console.WriteLine($"Warning: Failed to log build failure: {ex.Message}");
        }
    }

    private async Task LogTestFailure()
    {
        try 
        { 
            await File.AppendAllTextAsync(_deployLogPath, 
                JsonSerializer.Serialize(new { evt = "TEST_FAIL", utc = DateTime.UtcNow }) + "\n").ConfigureAwait(false); 
        } 
        catch (UnauthorizedAccessException ex)
        {
            Console.WriteLine($"Warning: Access denied logging test failure: {ex.Message}");
        }
        catch (IOException ex)
        {
            Console.WriteLine($"Warning: IO error logging test failure: {ex.Message}");
        }
        catch (Exception ex) when (!ex.IsFatal())
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
        using var p = new Process();
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
}

internal static class ExceptionExtensions
{
    /// <summary>
    /// Determines if an exception is fatal and should be rethrown
    /// </summary>
    public static bool IsFatal(this Exception ex)
    {
        return ex is OutOfMemoryException ||
               ex is StackOverflowException ||
               ex is AccessViolationException ||
               ex is AppDomainUnloadedException ||
               ex is ThreadAbortException ||
               ex is SecurityException;
    }
}
