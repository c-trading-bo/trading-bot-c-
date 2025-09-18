using System;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore;
using SupervisorAgent;
using System.Text.Json;
using BotCore.Models;
using BotCore.Risk;
using BotCore.Strategy;
using BotCore.Config;
using OrchestratorAgent.Infra;
using OrchestratorAgent.Ops;
using OrchestratorAgent.Intelligence;
using OrchestratorAgent.Critical;
using System.Linq;
using System.Net.Http.Json;
// using Dashboard; // Commented out - Dashboard module not available
using Microsoft.Extensions.DependencyInjection;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.Http;
using System.Reflection;
using OrchestratorAgent.ML;
using Trading.Safety;

namespace OrchestratorAgent
{
    // ❌ DISABLED - REPLACED BY UNIFIED ORCHESTRATOR SYSTEM ❌
    //
    // This OrchestratorAgent has been replaced by:
    // src/UnifiedOrchestrator/Program.cs
    //
    // The UnifiedOrchestrator provides:
    // - EnhancedTradingBrainIntegration with all ML/RL/Cloud services
    // - Neural UCB + CVaR-PPO + LSTM algorithms
    // - 30 GitHub workflows integration
    // - Production-grade error handling and monitoring
    //
    // To prevent conflicts, this system is DISABLED.
    // To run the active system: cd src/UnifiedOrchestrator && dotnet run
    
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("❌ OrchestratorAgent DISABLED");
            Console.WriteLine("🚀 Use UnifiedOrchestrator instead:");
            Console.WriteLine("   cd src/UnifiedOrchestrator && dotnet run");
            Console.WriteLine("");
            Console.WriteLine("⚠️  This system has been replaced by the enhanced multi-brain system.");
            Console.WriteLine("⚠️  Running this could conflict with your production trading bot.");
            
            await Task.Delay(3000).ConfigureAwait(false);
            return;
        }
    }
}
