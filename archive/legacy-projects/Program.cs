using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Trading.Strategies;
using System;
using System.IO;

namespace SimpleBot;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("❌ SimpleBot DISABLED");
        Console.WriteLine("🚀 Use UnifiedOrchestrator instead:");
        Console.WriteLine("   cd src/UnifiedOrchestrator && dotnet run");
        Console.WriteLine("");
        Console.WriteLine("⚠️  This simple bot has been replaced by the enhanced multi-brain system.");
        Console.WriteLine("⚠️  The UnifiedOrchestrator provides all SimpleBot functionality plus:");
        Console.WriteLine("   • Neural UCB + CVaR-PPO + LSTM algorithms");
        Console.WriteLine("   • 7 ML/RL/Cloud production services");
        Console.WriteLine("   • 30 GitHub workflows integration");
        Console.WriteLine("   • Enterprise-grade monitoring and error handling");
        
        await Task.Delay(3000);
        return;
    }
    }
}
