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
            var jwtEnv = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            var acctEnv = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID");

            if (string.IsNullOrWhiteSpace(jwtEnv) || !long.TryParse(acctEnv, out var accountId))
                throw new InvalidOperationException("Set TOPSTEPX_JWT and TOPSTEPX_ACCOUNT_ID before running the sim.");

            string jwt = jwtEnv!;
            var ct = new CancellationTokenSource(TimeSpan.FromSeconds(60)).Token;

            var agent = new UserHubAgentSim(logger);
            await agent.ConnectAsync(jwt, accountId, ct);

            Console.WriteLine("UserHub simulation complete.");
        }
    }
}
