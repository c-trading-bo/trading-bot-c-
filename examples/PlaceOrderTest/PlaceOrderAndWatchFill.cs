// ...existing using directives...

// ...existing using directives...

// Place this helper and call after all using directives

// ...existing code...

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

LoadEnvFile(".env.local");

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using BotCore;
using BotCore.Models;
using BotCore.Config;
using BotCore.Risk;

var username = Environment.GetEnvironmentVariable("TSX_USERNAME");
var apiKey = Environment.GetEnvironmentVariable("TSX_API_KEY");
if (string.IsNullOrWhiteSpace(username) || string.IsNullOrWhiteSpace(apiKey))
{
    Console.WriteLine("Missing TSX_USERNAME or TSX_API_KEY in environment or .env.local");
    return;
}
var authAgent = new BotCore.TopstepAuthAgent(username, apiKey);
var success = await authAgent.LoginAsync();
if (!success || string.IsNullOrWhiteSpace(authAgent.Token))
{
    Console.WriteLine("Login failed. Cannot proceed.");
    return;
}
var token = authAgent.Token;

// Market data agent
var contractId = "ESU5"; // TODO: fetch dynamically
var marketAgent = new BotCore.ReliableMarketDataAgent(token);
marketAgent.OnBar += bar => Console.WriteLine($"BAR: {bar?.Symbol} {bar?.Close:0.00}");
marketAgent.OnQuote += quote => Console.WriteLine($"QUOTE: {quote}");
marketAgent.OnTrade += trade => Console.WriteLine($"TRADE: {trade}");

// Order router agent
var orderAgent = new BotCore.OrderRouterAgent(token);
// Example usage: await orderAgent.PlaceOrderAsync(orderRequest, CancellationToken.None);

// Start market data stream (replace with actual contractId)
await marketAgent.StartAsync(contractId);

// ...existing code for strategy, risk, and order logic...

await Task.Delay(TimeSpan.FromMinutes(10));
