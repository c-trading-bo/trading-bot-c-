using Microsoft.AspNetCore.SignalR.Client;
using System;
using System.Net.Http;
using System.Text.Json;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore; // Ensure BotCore.UserHubAgent is used
using SupervisorAgent;           // Supervisor integration
using BotCore.Strategy;          // AllStrategies
using OrchestratorAgent;
using TopstepAuthAgent;          // TopstepAuthAgent
using BotCore.Models;
using BotCore.Risk;

namespace OrchestratorAgent
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            // Auto-load .env.local from working directory if present
            var envPath = Path.Combine(Directory.GetCurrentDirectory(), ".env.local");
            if (File.Exists(envPath))
            {
                foreach (var line in File.ReadAllLines(envPath))
                {
                    var trimmed = line.Trim();
                    if (string.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("#")) continue;
                    var parts = trimmed.Split('=', 2);
                    if (parts.Length == 2)
                    {
                        Environment.SetEnvironmentVariable(parts[0].Trim(), parts[1].Trim());
                    }
                }
            }

            string CurrentJwt() => Environment.GetEnvironmentVariable("TOPSTEPX_JWT") ?? string.Empty;

            void WireHandlers(MarketHubClient c, string root)
            {
                c.OnQuote += (cid, json) =>
                {
                    if (json.ValueKind == JsonValueKind.Object)
                    {
                        if (json.TryGetProperty("last", out var p) && p.ValueKind == JsonValueKind.Number)
                            Console.WriteLine($"[{root}] Quote last={p}");
                    }
                };
                c.OnTrade += (cid, json) => { };
                c.OnDepth += (cid, json) => { };
            }

            var marketClients = new List<MarketHubClient>();
            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

            string apiBase   = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
            string? userName = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME") ?? Environment.GetEnvironmentVariable("TSX_USERNAME");
            string? apiKey   = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY") ?? Environment.GetEnvironmentVariable("TSX_API_KEY");
            string? jwt      = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            string? acctStr  = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID") ?? Environment.GetEnvironmentVariable("TSX_ACCOUNT_ID") ?? Environment.GetEnvironmentVariable("ACCOUNT_ID");

            bool missing = false;
            if (string.IsNullOrWhiteSpace(acctStr))
            {
                Console.Error.WriteLine("ERROR: TOPSTEPX_ACCOUNT_ID missing. Tried TOPSTEPX_ACCOUNT_ID, TSX_ACCOUNT_ID, ACCOUNT_ID.");
                missing = true;
            }
            if (string.IsNullOrWhiteSpace(userName))
            {
                Console.Error.WriteLine("ERROR: TOPSTEPX_USERNAME/TSX_USERNAME missing.");
                missing = true;
            }
            if (string.IsNullOrWhiteSpace(apiKey))
            {
                Console.Error.WriteLine("ERROR: TOPSTEPX_API_KEY/TSX_API_KEY missing.");
                missing = true;
            }
            if (missing)
            {
                Console.Error.WriteLine("One or more required environment variables are missing. Starting in DEMO mode (no live connections).");
                Console.WriteLine("[DEMO] No credentials provided. Running offline. Press Ctrl+C to exit.");
                try { await Task.Delay(Timeout.Infinite, cts.Token); } catch (TaskCanceledException) { }
                return;
            }

            int accountId = 0;
            if (!int.TryParse(acctStr, out accountId) || accountId <= 0)
            {
                Console.Error.WriteLine("ERROR: TOPSTEPX_ACCOUNT_ID is not a valid integer.");
                Environment.Exit(1);
            }

            using var http = new HttpClient();
            using var apiHttp = new HttpClient();
            using var loggerFactory = LoggerFactory.Create(b => b
                .AddSimpleConsole(o => { o.TimestampFormat = "HH:mm:ss "; o.SingleLine = true; })
                .SetMinimumLevel(LogLevel.Information));

            var log = loggerFactory.CreateLogger("Orchestrator");
            var authLog = loggerFactory.CreateLogger<TopstepAuthAgent.TopstepAuthAgent>();

            var auth = new TopstepAuthAgent.TopstepAuthAgent(http, authLog, apiBase);
            if (string.IsNullOrWhiteSpace(jwt))
            {
                log.LogInformation("Logging in with loginKey…");
                jwt = await auth.GetJwtAsync(userName!, apiKey!, cts.Token);
            }
            else
            {
                log.LogInformation("Validating existing JWT…");
                try { jwt = await auth.ValidateAsync(jwt!, cts.Token); }
                catch { jwt = await auth.GetJwtAsync(userName!, apiKey!, cts.Token); }
            }
            Environment.SetEnvironmentVariable("TOPSTEPX_JWT", jwt);

            var policy = EvalPolicy.FromEnv();
            var pnl = new PnLTracker(policy);
            await using var userHub = new BotCore.UserHubAgent(loggerFactory.CreateLogger<BotCore.UserHubAgent>());
            await userHub.ConnectAsync(jwt!, accountId, cts.Token);
            var guard = new EvalGuard(policy, pnl);

            // Create marketHub after contracts are resolved
            var api = new ApiClient(apiHttp, loggerFactory.CreateLogger<ApiClient>(), apiBase);
            api.SetJwt(jwt!);

            var roots = new[] { "ES", "NQ" };
            var contractIds = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var r in roots)
            {
                try
                {
                    log.LogInformation("Resolving contract for symbol: {Root}", r);
                    var cid = await api.ResolveContractIdAsync(r, cts.Token);
                    contractIds[r] = cid;
                    log.LogInformation("Resolved {Root} -> {ContractId}", r, cid);
                }
                catch (Exception ex)
                {
                    log.LogWarning("Failed to resolve contract for {Root}: {Msg}", r, ex.Message);
                }
            }

            var status = new SupervisorAgent.StatusService(loggerFactory.CreateLogger<SupervisorAgent.StatusService>())
            {
                AccountId = accountId,
                Contracts = new() { ["ES"] = contractIds.ContainsKey("ES") ? contractIds["ES"] : "", ["NQ"] = contractIds.ContainsKey("NQ") ? contractIds["NQ"] : "" },
            };

            var marketHub = new MarketHubClient(loggerFactory.CreateLogger<MarketHubClient>(), CurrentJwt);
            WireHubState(userHub.Connection, marketHub.Connection, status, cts.Token);
            var supervisor = new SupervisorAgent.SupervisorAgent(
                loggerFactory.CreateLogger<SupervisorAgent.SupervisorAgent>(),
                http, apiBase, jwt!, accountId,
                marketHub,
                userHub,
                status,
                new SupervisorAgent.SupervisorAgent.Config
                {
                    LiveTrading = true,
                    BarSeconds = 60,
                    Symbols = new[] { "ES", "NQ" },
                    UseQuotes = true,
                    DefaultBracket = new SupervisorAgent.SupervisorAgent.BracketConfig
                    {
                        StopTicks = 12,
                        TargetTicks = 18,
                        BreakevenAfterTicks = 8,
                        TrailTicks = 6
                    }
                }
            );
            await supervisor.RunAsync(cts.Token);
            var botSupervisor = new OrchestratorAgent.BotSupervisor(
                loggerFactory.CreateLogger<OrchestratorAgent.BotSupervisor>(),
                http, apiBase, jwt!, accountId,
                marketHub,
                userHub,
                status,
                new OrchestratorAgent.BotSupervisor.Config
                {
                    LiveTrading = true,
                    BarSeconds = 60,
                    Symbols = new[] { "ES", "NQ" },
                    UseQuotes = true,
                    DefaultBracket = new OrchestratorAgent.BotSupervisor.BracketConfig
                    {
                        StopTicks = 12,
                        TargetTicks = 18,
                        BreakevenAfterTicks = 8,
                        TrailTicks = 6
                    }
                }
            );
            _ = botSupervisor.RunAsync(cts.Token);
            var bars = new BotCore.BarsRegistry(maxKeep: 1000);

            // All event wiring is now handled by WireHubState only

        }

        // WireHubState wires hub state and event subscriptions for orchestration
        public static void WireHubState(
            Microsoft.AspNetCore.SignalR.Client.HubConnection? userHubConn,
            Microsoft.AspNetCore.SignalR.Client.HubConnection? marketHubConn,
            SupervisorAgent.StatusService status,
            CancellationToken ct)
        {
            if (userHubConn != null)
            {
                userHubConn.On("GatewayUserOrder", (object[] args) =>
                {
                    if (args != null && args.Length > 0)
                    {
                        var json = System.Text.Json.JsonSerializer.Serialize(args[0]);
                        status.Set("order", json);
                    }
                });
                userHubConn.On("GatewayUserTrade", (object[] args) =>
                {
                    if (args != null && args.Length > 0)
                    {
                        var json = System.Text.Json.JsonSerializer.Serialize(args[0]);
                        status.Set("trade", json);
                    }
                });
            }
            if (marketHubConn != null)
            {
                marketHubConn.On("GatewayQuote", (object[] args) =>
                {
                    if (args != null && args.Length > 1)
                    {
                        var cid = args[0]?.ToString();
                        var json = System.Text.Json.JsonSerializer.Serialize(args[1]);
                        status.Set($"quote:{cid}", json ?? "");
                    }
                });
                marketHubConn.On("GatewayTrade", (object[] args) =>
                {
                    if (args != null && args.Length > 1)
                    {
                        var cid = args[0]?.ToString();
                        var json = System.Text.Json.JsonSerializer.Serialize(args[1]);
                        status.Set($"trade:{cid}", json ?? "");
                    }
                });
                marketHubConn.On("GatewayDepth", (object[] args) =>
                {
                    if (args != null && args.Length > 1)
                    {
                        var cid = args[0]?.ToString();
                        var json = System.Text.Json.JsonSerializer.Serialize(args[1]);
                        status.Set($"depth:{cid}", json ?? "");
                    }
                });
            }
        }
    }
}



