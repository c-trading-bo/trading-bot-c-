using System;
using System.Text.Json;
using System.Threading.Tasks;
using BotCore;

void LoadEnvFile(string path)
{
    if (!System.IO.File.Exists(path)) return;
    foreach (var line in System.IO.File.ReadAllLines(path))
    {
        var trimmed = line.Trim();
        if (string.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("#")) continue;
        var parts = trimmed.Split('=', 2);
        if (parts.Length == 2)
            Environment.SetEnvironmentVariable(parts[0], parts[1]);
    }
}

// Load environment variables from .env.local if present
LoadEnvFile(".env.local");

// Authenticate to get JWT
var username = Environment.GetEnvironmentVariable("TSX_USERNAME");
var apiKey = Environment.GetEnvironmentVariable("TSX_API_KEY");
if (string.IsNullOrWhiteSpace(username) || string.IsNullOrWhiteSpace(apiKey))
{
    Console.WriteLine("Missing TSX_USERNAME or TSX_API_KEY in environment or .env.local");
    return;
}

var authAgent = new BotCore.TopstepAuthAgent(username, apiKey);
var loginOk = await authAgent.LoginAsync();
if (!loginOk || string.IsNullOrWhiteSpace(authAgent.Token))
{
    Console.WriteLine("Login failed. Cannot proceed.");
    return;
}

var token = authAgent.Token;

// Stream market data briefly for a couple of contracts to verify connectivity
var envAccountId = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID");
if (string.IsNullOrWhiteSpace(envAccountId))
{
    Console.WriteLine("TOPSTEPX_ACCOUNT_ID not set in environment. Cannot attach account.");
    return;
}
Console.WriteLine($"Using accountId from env: {envAccountId}");

foreach (var contractId in new[] { "ESU5", "NQU5" })
{
    Console.WriteLine($"\n=== Testing contract: {contractId} ===");
    await using var md = new BotCore.MarketData.ReliableMarketDataAgent();

    try
    {
        using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromSeconds(30));
        await md.ConnectAsync(cts.Token);
        Console.WriteLine("Connected to market hub.");

        await md.AttachAccountAsync(envAccountId, cts.Token);
        Console.WriteLine("Attached account group.");

        await md.SubscribeBarsAsync(contractId, "1m", cts.Token);
        await md.SubscribeQuoteAsync(contractId, cts.Token);
        Console.WriteLine("Subscribed to bars and quotes.");

        var start = DateTime.UtcNow;
        var lastBars = -1;
        while ((DateTime.UtcNow - start).TotalSeconds < 20)
        {
            await Task.Delay(1000);
            if (md.BarsSeen != lastBars)
            {
                Console.WriteLine($"heartbeat: bars={md.BarsSeen} last={md.LastPrice}");
                lastBars = md.BarsSeen;
            }
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"Error streaming market data for {contractId}: {ex.Message}");
    }
}

Console.WriteLine("Done.");
