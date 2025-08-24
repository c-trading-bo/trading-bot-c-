using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace SimulationAgent
{
    class ProgramUserHubSim
    {
        static async Task Main(string[] args)
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
