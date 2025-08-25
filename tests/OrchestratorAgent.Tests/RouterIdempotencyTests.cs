using System;
using System.Net.Http;
using System.Threading;
using Microsoft.Extensions.Logging;
using OrchestratorAgent;
using BotCore;
using Xunit;

public class RouterIdempotencyTests
{
    [Fact]
    public async Task Router_DropsDuplicateCid()
    {
        // Ensure DRY-RUN
        Environment.SetEnvironmentVariable("LIVE_ORDERS", "0");

        using var http = new HttpClient { BaseAddress = new Uri("http://localhost") };
        var loggerFactory = LoggerFactory.Create(b => b.AddConsole().SetMinimumLevel(LogLevel.Critical));
        var log = loggerFactory.CreateLogger<OrchestratorAgent.OrderRouter>();

        var router = new OrchestratorAgent.OrderRouter(log, http, "http://localhost", "jwt", 123);

        var sig = new StrategySignal {
            Strategy = "S1",
            Symbol = "ES",
            Side = SignalSide.Long,
            LimitPrice = 5000m,
            Size = 1,
            ClientOrderId = "CID1"
        };

        var ok1 = await router.RouteAsync(sig, "CONTRACT", CancellationToken.None);
        var ok2 = await router.RouteAsync(sig, "CONTRACT", CancellationToken.None);

        Assert.True(ok1);
        Assert.False(ok2);
    }
}
