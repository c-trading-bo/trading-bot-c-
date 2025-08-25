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

            // TODO: Replace with your real JWT and accountId for testing
            string jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT") ?? "your-jwt-token-here";
            long accountId = long.TryParse(Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID"), out var id) ? id : 123456;
            var ct = new CancellationTokenSource(TimeSpan.FromSeconds(60)).Token;

            var agent = new UserHubAgentSim(logger);
            await agent.ConnectAsync(jwt, accountId, ct);

            Console.WriteLine("UserHub simulation complete.");
        }
    }
}
