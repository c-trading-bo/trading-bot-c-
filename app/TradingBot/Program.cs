using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using TradingBot.Core.Intelligence;
using TradingBot.Orchestrators;
using BotCore.Services;
using BotCore.ML;

namespace TradingBot.Production
{
    // ❌ DISABLED - REPLACED BY UNIFIED ORCHESTRATOR SYSTEM ❌
    //
    // This TradingBot app has been replaced by:
    // src/UnifiedOrchestrator/Program.cs
    //
    // The UnifiedOrchestrator provides ALL functionality from this app PLUS:
    // - EnhancedTradingBrainIntegration with Neural UCB + CVaR-PPO + LSTM
    // - All 7 ML/RL/Cloud production services
    // - 30 GitHub workflows integration for continuous model training
    // - Enterprise-grade error handling, monitoring, and configuration
    //
    // To prevent conflicts, this system is DISABLED.
    
    /// <summary>
    /// LEGACY Production Application - REPLACED by UnifiedOrchestrator
    /// </summary>
    class ProductionApp
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("❌ app/TradingBot DISABLED");
            Console.WriteLine("🚀 Use UnifiedOrchestrator instead:");
            Console.WriteLine("   cd src/UnifiedOrchestrator && dotnet run");
            Console.WriteLine("");
            Console.WriteLine("⚠️  This system has been replaced by the enhanced multi-brain system.");
            Console.WriteLine("⚠️  The UnifiedOrchestrator includes ALL algorithms from this app plus ML/RL/Cloud integration.");
            Console.WriteLine("⚠️  Running this could conflict with your production trading bot.");
            
            await Task.Delay(3000);
            return;
        }
    }
}
