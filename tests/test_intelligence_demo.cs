using Microsoft.Extensions.Logging;
using BotCore.Services;
using BotCore.Models;
using System.Text.Json;

// Simple test program to demonstrate Intelligence Service
var logger = LoggerFactory.Create(builder => builder.AddConsole()).CreateLogger<IntelligenceService>();

// Test with example signals file
var intelligenceService = new IntelligenceService(logger, "Intelligence/data/signals/example_latest.json");

Console.WriteLine("=== Intelligence Service Demo ===");
Console.WriteLine();

// Test availability
bool isAvailable = intelligenceService.IsIntelligenceAvailable();
Console.WriteLine($"Intelligence Available: {isAvailable}");

if (isAvailable)
{
    // Test loading intelligence
    var intelligence = await intelligenceService.GetLatestIntelligenceAsync();
    
    if (intelligence != null)
    {
        Console.WriteLine($"Date: {intelligence.Date}");
        Console.WriteLine($"Regime: {intelligence.Regime}");
        Console.WriteLine($"Model Confidence: {intelligence.ModelConfidence:P1}");
        Console.WriteLine($"Primary Bias: {intelligence.PrimaryBias}");
        Console.WriteLine($"News Intensity: {intelligence.NewsIntensity:F1}/100");
        Console.WriteLine($"Special Events: CPI={intelligence.IsCpiDay}, FOMC={intelligence.IsFomcDay}");
        Console.WriteLine($"Trade Setups: {intelligence.Setups.Count}");
        
        foreach (var setup in intelligence.Setups)
        {
            Console.WriteLine($"  - {setup.TimeWindow}: {setup.Direction} " +
                            $"(confidence: {setup.ConfidenceScore:P1}, " +
                            $"risk: {setup.SuggestedRiskMultiple:F1}x)");
            Console.WriteLine($"    Rationale: {setup.Rationale}");
        }
    }
    else
    {
        Console.WriteLine("Failed to load intelligence data");
    }
    
    // Test age calculation
    var age = intelligenceService.GetIntelligenceAge();
    if (age.HasValue)
    {
        Console.WriteLine($"Intelligence Age: {age.Value.TotalMinutes:F1} minutes");
    }
    
    // Test trade result logging
    Console.WriteLine();
    Console.WriteLine("Testing trade result logging...");
    await intelligenceService.LogTradeResultAsync("MES", 5425.25m, 5430.50m, 26.25m, intelligence);
    Console.WriteLine("Trade result logged successfully");
}
else
{
    Console.WriteLine("Intelligence not available - bot would continue with normal operation");
}

Console.WriteLine();
Console.WriteLine("=== Intelligence Service Demo Complete ===");