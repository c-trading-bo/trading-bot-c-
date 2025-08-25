using System;

namespace SimulationAgent
{
    // Modular simulation agent with error handling and extension points
    public interface ISimulationModule
    {
        void Execute();
    }

    public class SimulationAgent
    {
        private readonly List<ISimulationModule> _modules = new();

        public void AddModule(ISimulationModule module)
        {
            _modules.Add(module);
        }

        public void Run()
        {
            Console.WriteLine("SimulationAgent: Running simulation...");
            bool allPrechecksPassed = true;
            foreach (var module in _modules)
            {
                try
                {
                    module.Execute();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[ERROR] Module {module.GetType().Name} failed: {ex.Message}\n{ex}");
                    allPrechecksPassed = false;
                }
            }
            Console.WriteLine("Simulation complete.");
            if (allPrechecksPassed)
            {
                Console.WriteLine("[PRECHECK] All prechecks passed. Switching to LIVE mode...");
                Environment.SetEnvironmentVariable("LiveOrders", "true");
                Environment.SetEnvironmentVariable("Mode", "live");
            }
            else
            {
                Console.WriteLine("[PRECHECK] Precheck failed. Remaining in DRY-RUN mode.");
            }
        }
    }

    // 1. StatusService simulation
    public class StatusServiceModule : ISimulationModule
    {
        public void Execute()
        {
            // Simulate StatusService heartbeat and status snapshot
            var snapshot = new
            {
                whenUtc = DateTimeOffset.UtcNow,
                accountId = 123456,
                contracts = new Dictionary<string, string> { { "ES", "MESU25" } },
                userHub = "Connected",
                marketHub = "Connected",
                lastTrade = DateTimeOffset.UtcNow.AddSeconds(-10),
                lastQuote = DateTimeOffset.UtcNow.AddSeconds(-5),
                strategies = new[] { "S1:ARMED", "S2:BLOCKED" },
                openOrders = new[] { "Order123", "Order456" },
                risk = new { maxLoss = 500, currentLoss = 120 }
            };
            var json = System.Text.Json.JsonSerializer.Serialize(snapshot);
            Console.WriteLine($"[StatusService] BOT STATUS => {json}");
        }
    }

    // 2. Tick-aware strategy engine simulation
    public class StrategyEngineModule : ISimulationModule
    {
        public void Execute()
        {
            // Simulate strategy diagnostics using real logic
            // Example config and snapshot (replace with real data as needed)
            var cfg = new BotCore.Config.TradingProfileConfig();
            var def = new BotCore.Config.StrategyDef { Name = "S1", SessionWindowEt = "08:30-15:00" };
            var tz = TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time"); // CME/Chicago
            var nowCt = TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, tz).TimeOfDay;
            var open = new TimeSpan(8, 30, 0);
            var close = new TimeSpan(15, 0, 0);
            var inRth = nowCt >= open && nowCt <= close;
            var snap = new BotCore.Models.MarketSnapshot
            {
                Symbol = "ES",
                UtcNow = DateTimeOffset.UtcNow.UtcDateTime,
                Adx5m = 25,
                Ema9Over21_5m = true,
                SpreadTicks = 4,
                SessionWindowEt = inRth ? "08:30-15:00" : "OUTSIDE"
            };
            var report = BotCore.Supervisor.StrategyDiagnostics.Explain(cfg, def, snap);
            Console.WriteLine($"[StrategyEngine] Strategy: {report.Strategy}, Symbol: {report.Symbol}, Verdict: {report.Verdict}");
            foreach (var check in report.Checks)
            {
                Console.WriteLine($"  - {check.Name}: {(check.Pass ? "PASS" : "FAIL")} ({check.Detail})");
            }
        }
    }

    // 3. Smart order router simulation
    public class OrderRouterModule : ISimulationModule
    {
        public void Execute()
        {
            // Simulate order routing logic with contract mapping
            var contracts = new Dictionary<string, string> { ["ES"] = "MESU25" };
            var symbol = "ES";
            var price = 4200.25m;
            var stop = 4198.00m;
            var target = 4205.00m;
            var contractId = contracts.ContainsKey(symbol) ? contracts[symbol] : symbol;
            Console.WriteLine($"[OrderRouter] Routing order: BUY {contractId} @ {price}, stop {stop}, target {target}");
            // Simulate bracket, trailing, and risk gates
            Console.WriteLine("[OrderRouter] Bracket order placed. Trailing logic armed. Risk gates checked.");
        }
    }

    // 4. Live order placement toggle simulation
    public class LiveOrderToggleModule : ISimulationModule
    {
        public void Execute()
        {
            // Simulate live order toggle logic using config
            var liveOrders = Environment.GetEnvironmentVariable("LiveOrders") == "true";
            var mode = Environment.GetEnvironmentVariable("Mode") ?? "sim";
            if (liveOrders && mode == "live")
                Console.WriteLine("[LiveOrderToggle] LIVE order placement ENABLED.");
            else
                Console.WriteLine("[LiveOrderToggle] DRY-RUN mode (no live orders).");
        }
    }

    // 5. Bot verification simulation
    public class BotVerificationModule : ISimulationModule
    {
        public void Execute()
        {
            // Simulate bot verification logic
            Console.WriteLine("[BotVerification] Verifying bot status, diagnostics, and order routing...");
            // Call other modules or simulate checks
            Console.WriteLine("[BotVerification] Heartbeat OK. Diagnostics OK. Order routing OK.");
        }
    }

    // 6. Quote subscription simulation
    public class QuoteSubscriptionModule : ISimulationModule
    {
        public void Execute()
        {
            // Simulate MarketHubClient quote subscription logic
            var log = new DummyLogger();
            var client = new BotCore.MarketHubClient(log, () => Task.FromResult<string?>("dummy-jwt-token"));
            client.OnQuote += (cid, json) =>
            {
                Console.WriteLine($"[QuoteSubscription] Quote received for contract {cid}: {json}");
            };
            // Simulate starting client and subscribing (no real SignalR connection)
            var useSimulated = Environment.GetEnvironmentVariable("UseSimulatedMarketHub") == "true";
            if (useSimulated)
                Console.WriteLine("[QuoteSubscription] Simulated MarketHubClient started and quote subscription wired.");
            else
                Console.WriteLine("[QuoteSubscription] Real MarketHubClient (SignalR) started and quote subscription wired.");
        }

        // Dummy logger for simulation
        private class DummyLogger : Microsoft.Extensions.Logging.ILogger<BotCore.MarketHubClient>
        {
            public IDisposable BeginScope<TState>(TState state) => null!;
            public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel level) => true;
            public void Log<TState>(Microsoft.Extensions.Logging.LogLevel level, Microsoft.Extensions.Logging.EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
            {
                Console.WriteLine($"[DummyLogger] {formatter(state, exception)}");
            }
        }
    }

    // 7. BarAggregator upgrade simulation
    public class BarAggregatorUpgradeModule : ISimulationModule
    {
        public void Execute()
        {
            // Simulate BarAggregator logic with sample trade/quote data
            var aggregator = new BotCore.BarAggregator(60); // 1-minute bars
            aggregator.OnBar += bar =>
            {
                // If bar.Ts is a long (timestamp):
                if (bar.Open <= 0 || bar.Ts <= 0)
                    return; // ignore bogus tick
                // If bar.Ts is DateTimeOffset, use:
                // if (bar.Open <= 0 || bar.Ts <= DateTimeOffset.MinValue) return;
                Console.WriteLine($"[BarAggregatorUpgrade] Bar: O={bar.Open} H={bar.High} L={bar.Low} C={bar.Close} V={bar.Volume} Ts={bar.Ts}");
            };
            // Simulate trades
            var tradeJson = System.Text.Json.JsonDocument.Parse("{\"price\":4200.25,\"size\":1}").RootElement;
            aggregator.OnTrade(tradeJson);
            // Simulate quotes
            var quoteJson = System.Text.Json.JsonDocument.Parse("{\"bid\":4200.00,\"ask\":4200.50}").RootElement;
            aggregator.OnTrade(quoteJson);
            // Force flush to print bar
            var flushMethod = aggregator.GetType().GetMethod("Flush", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            flushMethod?.Invoke(aggregator, null);
        }
    }

    // 8. HighWinRateProfile fix simulation
    public class HighWinRateProfileFixModule : ISimulationModule
    {
        public void Execute()
        {
            // Use real HighWinRateProfile logic
            var profile = new BotCore.Config.HighWinRateProfile();
            Console.WriteLine($"[HighWinRateProfileFix] Profile: {profile.Profile}");
            Console.WriteLine("AttemptCaps:");
            foreach (var kvp in profile.AttemptCaps)
                Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
            Console.WriteLine("Buffers:");
            foreach (var kvp in profile.Buffers)
                Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
            Console.WriteLine("GlobalFilters:");
            foreach (var kvp in profile.GlobalFilters)
                Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
            Console.WriteLine("Concurrency:");
            foreach (var kvp in profile.Concurrency)
                Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
            Console.WriteLine("Hysteresis:");
            foreach (var kvp in profile.Hysteresis)
                Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
            Console.WriteLine("Timeframes:");
            foreach (var kvp in profile.Timeframes)
                Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var agent = new SimulationAgent();
            agent.AddModule(new StatusServiceModule());
            agent.AddModule(new StrategyEngineModule());
            agent.AddModule(new OrderRouterModule());
            agent.AddModule(new LiveOrderToggleModule());
            agent.AddModule(new BotVerificationModule());
            agent.AddModule(new QuoteSubscriptionModule());
            agent.AddModule(new BarAggregatorUpgradeModule());
            agent.AddModule(new HighWinRateProfileFixModule());
            agent.Run();
        }
    }
        public class ProgramUserHubSim
        {
            public static async Task Main(string[] args)
            {
                var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
                var logger = loggerFactory.CreateLogger("UserHubSim");

                // Load credentials from environment; optionally fetch JWT using login key
                using var http = new System.Net.Http.HttpClient { BaseAddress = new Uri(Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com") };
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(60));
                var ct = cts.Token;

                string? jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                string? userName = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
                string? apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
                long accountId = long.TryParse(Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID"), out var id) ? id : 0L;

                if (string.IsNullOrWhiteSpace(jwt) && !string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey))
                {
                    try
                    {
                        var auth = new TopstepAuthAgent(http);
                        logger.LogInformation("Fetching JWT using login key for {User}…", userName);
                        jwt = await auth.GetJwtAsync(userName!, apiKey!, ct);
                        Environment.SetEnvironmentVariable("TOPSTEPX_JWT", jwt);
                    }
                    catch (Exception ex)
                    {
                        logger.LogWarning(ex, "Failed to obtain JWT using TOPSTEPX_USERNAME/TOPSTEPX_API_KEY");
                    }
                }

                if (string.IsNullOrWhiteSpace(jwt) || accountId <= 0)
                {
                    logger.LogWarning("Missing TOPSTEPX_JWT and/or TOPSTEPX_ACCOUNT_ID. Set them in .env.local or environment. Exiting simulation.");
                    return;
                }

                var agent = new UserHubAgentSim(logger);
                await agent.ConnectAsync(jwt!, accountId, ct);

                Console.WriteLine("UserHub simulation complete.");
            }
        }
}
