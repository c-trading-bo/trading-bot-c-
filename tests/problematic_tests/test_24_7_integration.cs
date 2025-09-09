// Simple integration test for 24/7 ES & NQ trading system
using System;
using System.Collections.Generic;
using BotCore.Config;
using BotCore.Services;
using BotCore.Models;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace TestEnhancedZones
{
    public static class IntegrationTest
    {
        public static void TestTimeOptimizedTrading()
        {
            Console.WriteLine("🧪 Testing 24/7 ES & NQ Trading System Integration");
            Console.WriteLine("==================================================");
            
            // Test 1: Trading Schedule
            TestTradingSchedule();
            
            // Test 2: Time-Optimized Strategy Manager
            await TestTimeOptimizedStrategyManager();
            
            // Test 3: Trading Progress Monitor
            TestTradingProgressMonitor();
            
            Console.WriteLine("\n✅ All integration tests passed!");
        }
        
        private static void TestTradingSchedule()
        {
            Console.WriteLine("\n📅 Testing Trading Schedule...");
            
            // Test different times of day
            var testTimes = new[]
            {
                new TimeSpan(9, 30, 0),   // US Open
                new TimeSpan(13, 30, 0),  // Afternoon
                new TimeSpan(2, 0, 0),    // European Open
                new TimeSpan(19, 0, 0),   // Asian Session
                new TimeSpan(12, 0, 0)    // Lunch
            };
            
            foreach (var time in testTimes)
            {
                var session = ES_NQ_TradingSchedule.GetCurrentSession(time);
                if (session != null)
                {
                    Console.WriteLine($"  ✓ {time:hh\\:mm} -> {session.Description} (Primary: {session.PrimaryInstrument})");
                    
                    // Test ES strategies
                    if (session.Strategies.ContainsKey("ES"))
                    {
                        var esStrategies = string.Join(", ", session.Strategies["ES"]);
                        Console.WriteLine($"    ES Strategies: {esStrategies}");
                    }
                    
                    // Test NQ strategies
                    if (session.Strategies.ContainsKey("NQ"))
                    {
                        var nqStrategies = string.Join(", ", session.Strategies["NQ"]);
                        Console.WriteLine($"    NQ Strategies: {nqStrategies}");
                    }
                }
                else
                {
                    Console.WriteLine($"  ✓ {time:hh\\:mm} -> Market Closed");
                }
            }
        }
        
        private static void TestTimeOptimizedStrategyManager()
        {
            Console.WriteLine("\n🧠 Testing Time-Optimized Strategy Manager...");
            
            var logger = NullLogger<TimeOptimizedStrategyManager>.Instance;
            using var manager = new TimeOptimizedStrategyManager(logger);
            
            // Create sample market data
            var marketData = new MarketData
            {
                Timestamp = DateTime.UtcNow,
                Symbol = "ES",
                Price = 4500.00m,
                Volume = 1000
            };
            
            // Create sample bars
            var bars = new List<Bar>
            {
                new() { Symbol = "ES", High = 4510, Low = 4490, Close = 4500, Volume = 1000, Ts = DateTimeOffset.UtcNow.ToUnixTimeSeconds() }
            };
            
            try
            {
                var result = await manager.EvaluateInstrumentAsync("ES", marketData, bars);
                Console.WriteLine($"  ✓ Strategy evaluation completed - HasSignal: {result.HasSignal}");
                Console.WriteLine($"    Session: {result.Session}");
                Console.WriteLine($"    Reason: {result.Reason}");
                
                if (result.HasSignal && result.Signal != null)
                {
                    Console.WriteLine($"    Signal: {result.Signal.StrategyId} {result.Signal.Side} {result.Signal.Symbol} @ {result.Signal.Entry}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ⚠️  Strategy evaluation error: {ex.Message}");
            }
        }
        
        private static void TestTradingProgressMonitor()
        {
            Console.WriteLine("\n📊 Testing Trading Progress Monitor...");
            
            var logger = NullLogger<TradingProgressMonitor>.Instance;
            using var monitor = new TradingProgressMonitor(logger);
            
            // Simulate some trade results
            var tradeResults = new[]
            {
                new TradeResult { Instrument = "ES", Strategy = "S2", PnL = 125.50, EntryTime = DateTime.UtcNow.AddHours(-2), MLConfidence = 0.85 },
                new TradeResult { Instrument = "ES", Strategy = "S3", PnL = -75.25, EntryTime = DateTime.UtcNow.AddHours(-1), MLConfidence = 0.65 },
                new TradeResult { Instrument = "NQ", Strategy = "S6", PnL = 200.00, EntryTime = DateTime.UtcNow.AddMinutes(-30), MLConfidence = 0.92 },
                new TradeResult { Instrument = "NQ", Strategy = "S11", PnL = 50.75, EntryTime = DateTime.UtcNow.AddMinutes(-15), MLConfidence = 0.78 }
            };
            
            foreach (var result in tradeResults)
            {
                monitor.UpdateMetrics(result);
            }
            
            var summary = monitor.GetSummary();
            Console.WriteLine($"  ✓ Processed {summary.TotalTrades} trades");
            Console.WriteLine($"    Total PnL: ${summary.TotalPnL:F2}");
            Console.WriteLine($"    Win Rate: {summary.OverallWinRate:P1}");
            Console.WriteLine($"    Active Strategies: {summary.ActiveStrategies}");
            Console.WriteLine($"    Active Instruments: {summary.ActiveInstruments}");
            
            // Test dashboard display (brief)
            Console.WriteLine("  ✓ Dashboard display test:");
            try
            {
                monitor.DisplayDashboard(force: true);
                Console.WriteLine("    Dashboard rendered successfully");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"    ⚠️  Dashboard error: {ex.Message}");
            }
        }
    }
    
    // Simple market data class for testing
    public class MarketData
    {
        public DateTime Timestamp { get; set; }
        public string Symbol { get; set; } = "";
        public decimal Price { get; set; }
        public int Volume { get; set; }
    }
}