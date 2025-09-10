/*
 * Test C# bot intelligence integration
 * This script verifies that the C# bot can consume intelligence data
 */

extern alias BotCoreTest;

using System;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCoreTest::BotCore.Services;

namespace TestBotIntelligence
{
    public class BotIntelligenceTestProgram
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("🧪 Testing C# Bot Intelligence Integration");
            Console.WriteLine("==================================================");

            // Create logger
            using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            var logger = loggerFactory.CreateLogger<IntelligenceService>();

            try
            {
                // Test Intelligence Service
                Console.WriteLine("\n1. Testing IntelligenceService...");
                
                var intelligenceService = new IntelligenceService(logger, "Intelligence/data/signals/latest.json");
                
                // Test intelligence loading
                var intelligence = await intelligenceService.GetLatestIntelligenceAsync();
                
                if (intelligence != null)
                {
                    Console.WriteLine($"   ✅ Intelligence loaded successfully!");
                    Console.WriteLine($"   📊 Market Regime: {intelligence.Regime}");
                    Console.WriteLine($"   🎯 Model Confidence: {intelligence.ModelConfidence:P0}");
                    Console.WriteLine($"   📈 Primary Bias: {intelligence.PrimaryBias}");
                    Console.WriteLine($"   📰 News Intensity: {intelligence.NewsIntensity:F1}");
                    Console.WriteLine($"   📅 Date: {intelligence.Date}");
                    Console.WriteLine($"   🎬 Setups: {intelligence.Setups.Count}");
                    
                    if (intelligence.IsFomcDay)
                        Console.WriteLine("   🚨 FOMC Day detected!");
                    if (intelligence.IsCpiDay)
                        Console.WriteLine("   📊 CPI Day detected!");
                }
                else
                {
                    Console.WriteLine("   ❌ Failed to load intelligence data");
                    return;
                }

                // Test intelligence methods
                Console.WriteLine("\n2. Testing intelligence decision methods...");
                
                var shouldTrade = intelligenceService.ShouldTrade(intelligence);
                var positionMultiplier = intelligenceService.GetPositionSizeMultiplier(intelligence);
                var stopMultiplier = intelligenceService.GetStopLossMultiplier(intelligence);
                var targetMultiplier = intelligenceService.GetTakeProfitMultiplier(intelligence);
                var preferredStrategy = intelligenceService.GetPreferredStrategy(intelligence);
                var isHighVol = intelligenceService.IsHighVolatilityEvent(intelligence);
                
                Console.WriteLine($"   📊 Should Trade: {shouldTrade}");
                Console.WriteLine($"   📏 Position Size Multiplier: {positionMultiplier:F2}x");
                Console.WriteLine($"   🛑 Stop Loss Multiplier: {stopMultiplier:F2}x");
                Console.WriteLine($"   🎯 Take Profit Multiplier: {targetMultiplier:F2}x");
                Console.WriteLine($"   🎲 Preferred Strategy: {preferredStrategy}");
                Console.WriteLine($"   ⚡ High Volatility Event: {isHighVol}");

                // Test Zone Service
                Console.WriteLine("\n3. Testing ZoneService...");
                
                var zoneLogger = loggerFactory.CreateLogger<ZoneService>();
                var zoneService = new ZoneService(zoneLogger, "Intelligence/data/zones/latest_zones.json");
                
                var zones = await zoneService.GetLatestZonesAsync("ES");
                
                if (zones != null)
                {
                    Console.WriteLine($"   ✅ Zone data loaded successfully!");
                    Console.WriteLine($"   📍 Symbol: {zones.Symbol}");
                    Console.WriteLine($"   💰 Current Price: {zones.CurrentPrice:F2}");
                    Console.WriteLine($"   📈 Supply Zones: {zones.SupplyZones.Count}");
                    Console.WriteLine($"   📉 Demand Zones: {zones.DemandZones.Count}");
                    Console.WriteLine($"   🎯 POC: {zones.POC:F2}");
                    
                    // Test zone methods
                    var currentPrice = zones.CurrentPrice;
                    var nearestSupport = zoneService.GetNearestSupport("ES", currentPrice);
                    var nearestResistance = zoneService.GetNearestResistance("ES", currentPrice);
                    var optimalStopLong = zoneService.GetOptimalStopLevel("ES", currentPrice, true);
                    var optimalTargetLong = zoneService.GetOptimalTargetLevel("ES", currentPrice, true);
                    var zoneAdjustedSize = zoneService.GetZoneBasedPositionSize("ES", 2, currentPrice, true);
                    var isNearZone = zoneService.IsNearZone("ES", currentPrice, 0.005m);
                    
                    Console.WriteLine($"   📉 Nearest Support: {nearestSupport:F2}");
                    Console.WriteLine($"   📈 Nearest Resistance: {nearestResistance:F2}");
                    Console.WriteLine($"   🛑 Optimal Stop (Long): {optimalStopLong:F2}");
                    Console.WriteLine($"   🎯 Optimal Target (Long): {optimalTargetLong:F2}");
                    Console.WriteLine($"   📏 Zone-Adjusted Size: {zoneAdjustedSize:F1}");
                    Console.WriteLine($"   📍 Is Near Zone: {isNearZone}");
                }
                else
                {
                    Console.WriteLine("   ❌ Failed to load zone data");
                    return;
                }

                // Test file availability  
                Console.WriteLine("\n4. Testing intelligence file availability...");
                
                var isIntelligenceAvailable = intelligenceService.IsIntelligenceAvailable();
                var intelligenceAge = intelligenceService.GetIntelligenceAge();
                
                Console.WriteLine($"   📁 Intelligence Available: {isIntelligenceAvailable}");
                if (intelligenceAge.HasValue)
                {
                    Console.WriteLine($"   ⏰ Intelligence Age: {intelligenceAge.Value.TotalMinutes:F1} minutes");
                }

                // Test logging trade result
                Console.WriteLine("\n5. Testing trade result logging...");
                
                await intelligenceService.LogTradeResultAsync(
                    "ES", 
                    4500.00m, 
                    4510.00m, 
                    250.00m, 
                    intelligence
                );
                
                Console.WriteLine("   ✅ Trade result logged successfully");

                // Summary
                Console.WriteLine("\n" + new string('=', 50));
                Console.WriteLine("🎉 C# Bot Intelligence Integration Test: PASSED");
                Console.WriteLine("   ✅ IntelligenceService working correctly");
                Console.WriteLine("   ✅ ZoneService working correctly");
                Console.WriteLine("   ✅ All intelligence methods functional");
                Console.WriteLine("   ✅ Trade logging working");
                Console.WriteLine("\n🤖 The C# bot is ready to trade with intelligence!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n❌ Test failed with exception: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                Environment.Exit(1);
            }
        }
    }
}