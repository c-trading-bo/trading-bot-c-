using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using BotCore.ML;
using System;
using System.IO;
using System.Threading.Tasks;

namespace TestMultiTimeframe;

/// <summary>
/// Test program to verify Multi-Timeframe Training Pipeline works in real-time.
/// This directly tests the components without going through the full Lab Mode orchestrator.
/// </summary>
class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║   MULTI-TIMEFRAME TRAINING PIPELINE - REAL-TIME TEST          ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");
        Console.WriteLine();
        
        // Build service provider with all multi-timeframe components
        var services = new ServiceCollection();
        
        // Add logging
        services.AddLogging(builder =>
        {
            builder.AddConsole();
            builder.SetMinimumLevel(LogLevel.Information);
        });
        
        // Register multi-timeframe components
        services.AddSingleton<MultiTimeframeFeatureExtractor>();
        services.AddSingleton<MultiTimeframeDataLoader>();
        services.AddSingleton<MultiTimeframeDataAssembler>();
        services.AddSingleton<MultiTimeframeBatchCreator>();
        services.AddSingleton<MultiTimeframeTrainingPipeline>();
        
        var serviceProvider = services.BuildServiceProvider();
        
        try
        {
            // Get the pipeline
            var pipeline = serviceProvider.GetRequiredService<MultiTimeframeTrainingPipeline>();
            
            Console.WriteLine("✅ Multi-Timeframe components initialized successfully");
            Console.WriteLine();
            
            // Test with ES
            Console.WriteLine("📊 Testing ES symbol...");
            Console.WriteLine("─────────────────────────────────────────────────────────────");
            
            var esData = await pipeline.PrepareTrainingDataAsync(
                symbol: "ES",
                trainRatio: 0.67,
                valRatio: 0.17,
                batchSize: 32,
                shuffle: true
            );
            
            Console.WriteLine($"✅ ES Data Prepared:");
            Console.WriteLine($"   • Total samples: {esData.Statistics.TotalSamples}");
            Console.WriteLine($"   • Train: {esData.TrainBatches.Count} batches ({esData.Statistics.TrainSamples} samples)");
            Console.WriteLine($"   • Val: {esData.ValidationBatches.Count} batches ({esData.Statistics.ValidationSamples} samples)");
            Console.WriteLine($"   • Test: {esData.TestBatches.Count} batches ({esData.Statistics.TestSamples} samples)");
            Console.WriteLine($"   • Features: {esData.Statistics.NumFeatures5m} (5m) + {esData.Statistics.NumFeatures1m} (1m) = {esData.Statistics.TotalFeatures}");
            Console.WriteLine($"   • Date range: {esData.Statistics.TrainDateRange.start:yyyy-MM-dd} to {esData.Statistics.TestDateRange.end:yyyy-MM-dd}");
            Console.WriteLine($"   • Feature version: {esData.FeatureVersionHash}");
            Console.WriteLine();
            
            // Test with NQ
            Console.WriteLine("📊 Testing NQ symbol...");
            Console.WriteLine("─────────────────────────────────────────────────────────────");
            
            var nqData = await pipeline.PrepareTrainingDataAsync(
                symbol: "NQ",
                trainRatio: 0.67,
                valRatio: 0.17,
                batchSize: 32,
                shuffle: true
            );
            
            Console.WriteLine($"✅ NQ Data Prepared:");
            Console.WriteLine($"   • Total samples: {nqData.Statistics.TotalSamples}");
            Console.WriteLine($"   • Train: {nqData.TrainBatches.Count} batches ({nqData.Statistics.TrainSamples} samples)");
            Console.WriteLine($"   • Val: {nqData.ValidationBatches.Count} batches ({nqData.Statistics.ValidationSamples} samples)");
            Console.WriteLine($"   • Test: {nqData.TestBatches.Count} batches ({nqData.Statistics.TestSamples} samples)");
            Console.WriteLine();
            
            // Summary
            var totalSamples = esData.Statistics.TotalSamples + nqData.Statistics.TotalSamples;
            var totalBatches = esData.TrainBatches.Count + esData.ValidationBatches.Count + esData.TestBatches.Count +
                             nqData.TrainBatches.Count + nqData.ValidationBatches.Count + nqData.TestBatches.Count;
            
            Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║                      SUMMARY                                   ║");
            Console.WriteLine("╠════════════════════════════════════════════════════════════════╣");
            Console.WriteLine($"║  Total synchronized samples: {totalSamples,-31} ║");
            Console.WriteLine($"║  Total batches created: {totalBatches,-36} ║");
            Console.WriteLine($"║  Status: ✅ ALL COMPONENTS WORKING                            ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");
            Console.WriteLine();
            
            return 0;
        }
        catch (Exception ex)
        {
            Console.WriteLine();
            Console.WriteLine("❌ ERROR: " + ex.Message);
            Console.WriteLine();
            Console.WriteLine("Stack trace:");
            Console.WriteLine(ex.StackTrace);
            return 1;
        }
    }
}
