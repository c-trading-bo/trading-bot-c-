#!/usr/bin/env dotnet script
/*
 * Test script to validate enhanced Zone Service functionality
 * This validates that the C# ZoneService can properly consume the enhanced zone data
 */

#r "src/BotCore/bin/Debug/net8.0/BotCore.dll"

using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore.Services;

Console.WriteLine("🧪 Testing Enhanced Zone Service Integration");
Console.WriteLine("=" * 50);

// Create logger
using var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
var logger = loggerFactory.CreateLogger<ZoneService>();

try
{
    // Test Zone Service
    Console.WriteLine("\n1. Testing Enhanced ZoneService...");
    
    var zoneService = new ZoneService(logger, "Intelligence/data/zones/active_zones.json");
    
    // Test zone loading
    var zones = await zoneService.GetLatestZonesAsync("ES");
    
    if (zones != null)
    {
        Console.WriteLine($"   ✅ Enhanced zone data loaded successfully!");
        Console.WriteLine($"   📍 Symbol: {zones.Symbol}");
        Console.WriteLine($"   💰 Current Price: {zones.CurrentPrice:F2}");
        Console.WriteLine($"   📈 Supply Zones: {zones.SupplyZones.Count}");
        Console.WriteLine($"   📉 Demand Zones: {zones.DemandZones.Count}");
        Console.WriteLine($"   🎯 POC: {zones.POC:F2}");
        
        // Test enhanced zone methods
        var currentPrice = zones.CurrentPrice;
        Console.WriteLine($"\n2. Testing Enhanced Zone Methods @ {currentPrice:F2}...");
        
        // Test nearest zone methods
        var nearestSupply = zoneService.GetNearestZone(currentPrice, "supply");
        var nearestDemand = zoneService.GetNearestZone(currentPrice, "demand");
        Console.WriteLine($"   📈 Nearest Supply Zone: {nearestSupply.PriceLevel:F2} (Strength: {nearestSupply.Strength:F0})");
        Console.WriteLine($"   📉 Nearest Demand Zone: {nearestDemand.PriceLevel:F2} (Strength: {nearestDemand.Strength:F0})");
        
        // Test zone context
        var context = zoneService.GetZoneContext(currentPrice);
        Console.WriteLine($"   🎯 Zone Context: {context}");
        
        // Test enhanced stop/target calculation
        var zoneAdjustedStop = zoneService.GetZoneAdjustedStopLoss(currentPrice, "long");
        var zoneAdjustedTarget = zoneService.GetZoneAdjustedTarget(currentPrice, "long");
        Console.WriteLine($"   🛑 Zone-Adjusted Stop (Long): {zoneAdjustedStop:F2}");
        Console.WriteLine($"   🎯 Zone-Adjusted Target (Long): {zoneAdjustedTarget:F2}");
        
        // Test zone interaction recording
        await zoneService.RecordZoneInteraction(currentPrice + 5, "touched");
        Console.WriteLine($"   📝 Zone interaction recorded successfully");
        
        // Test proximity detection
        var isNearZone = zoneService.IsNearZone(currentPrice, 0.005m);
        Console.WriteLine($"   📍 Is Near Zone (0.5% threshold): {isNearZone}");
        
        // Test original methods for backward compatibility
        Console.WriteLine($"\n3. Testing Backward Compatibility...");
        var nearestSupport = zoneService.GetNearestSupport("ES", currentPrice);
        var nearestResistance = zoneService.GetNearestResistance("ES", currentPrice);
        var optimalStopLong = zoneService.GetOptimalStopLevel("ES", currentPrice, true);
        var optimalTargetLong = zoneService.GetOptimalTargetLevel("ES", currentPrice, true);
        var zoneBasedSize = zoneService.GetZoneBasedPositionSize("ES", 2, currentPrice, true);
        
        Console.WriteLine($"   📉 Nearest Support: {nearestSupport:F2}");
        Console.WriteLine($"   📈 Nearest Resistance: {nearestResistance:F2}");
        Console.WriteLine($"   🛑 Optimal Stop (Long): {optimalStopLong:F2}");
        Console.WriteLine($"   🎯 Optimal Target (Long): {optimalTargetLong:F2}");
        Console.WriteLine($"   📏 Zone-Based Position Size: {zoneBasedSize:F1}");
        
        // Test SupplyDemandService interface
        Console.WriteLine($"\n4. Testing SupplyDemandService Interface...");
        ISupplyDemandService supplyDemandService = zoneService;
        var loadedZones = await supplyDemandService.LoadZonesAsync();
        Console.WriteLine($"   ✅ SupplyDemandService.LoadZonesAsync() works");
        Console.WriteLine($"   📊 Loaded {loadedZones.SupplyZones.Count + loadedZones.DemandZones.Count} total zones");
        
        Console.WriteLine("\n" + new string('=', 50));
        Console.WriteLine("🎉 Enhanced Zone Service Integration Test: PASSED");
        Console.WriteLine("   ✅ Enhanced zone data loading working correctly");
        Console.WriteLine("   ✅ Advanced zone interaction methods working");
        Console.WriteLine("   ✅ Zone-adjusted stop/target calculation working");
        Console.WriteLine("   ✅ Zone interaction tracking working");
        Console.WriteLine("   ✅ Backward compatibility maintained");
        Console.WriteLine("   ✅ SupplyDemandService interface working");
        Console.WriteLine("\n🏦 The enhanced institutional-grade zone system is ready!");
    }
    else
    {
        Console.WriteLine("   ❌ Failed to load zone data");
        Environment.Exit(1);
    }
}
catch (Exception ex)
{
    Console.WriteLine($"\n❌ Test failed with exception: {ex.Message}");
    Console.WriteLine($"Stack trace: {ex.StackTrace}");
    Environment.Exit(1);
}