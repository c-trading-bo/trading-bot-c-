using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using Dashboard;

namespace OfflineDashboardTest
{
    public class Program
    {
        public static async Task Main(string[] args)
        {
            Console.WriteLine("🎯 Starting Offline Dashboard Test...");
            
            var builder = WebApplication.CreateBuilder(args);
            
            // Configure logging
            builder.Logging.ClearProviders();
            builder.Logging.AddConsole();
            
            // Configure URLs
            builder.WebHost.UseUrls("http://localhost:5050");
            
            // Add services
            builder.Services.AddSingleton<RealtimeHub>(sp =>
            {
                var logger = sp.GetRequiredService<ILogger<RealtimeHub>>();
                return new RealtimeHub(logger, GetMockMetrics);
            });
            
            var app = builder.Build();
            
            // Configure the app
            app.UseDefaultFiles();
            app.UseStaticFiles();
            
            var hub = app.Services.GetRequiredService<RealtimeHub>();
            app.MapDashboard(hub);
            
            Console.WriteLine("✅ Dashboard started on http://localhost:5050");
            Console.WriteLine("📊 Visit http://localhost:5050/dashboard to view the dashboard");
            Console.WriteLine("🔄 Press Ctrl+C to stop");
            
            await app.RunAsync();
        }
        
        private static MetricsSnapshot GetMockMetrics()
        {
            return new MetricsSnapshot(
                accountId: 12345,
                mode: "PAPER",
                realized: 125.50m,
                unrealized: -45.75m,
                day: 79.75m,
                maxDailyLoss: -2000m,
                remaining: 1920.25m,
                userHub: "Connected",
                marketHub: "Connected", 
                localTime: DateTime.Now,
                positions: new List<PositionChip>
                {
                    new("ES", 2, 5000.25m, 5010.00m, 97.50m, 0m),
                    new("NQ", -1, 18000.50m, 17985.25m, 60.50m, 0m)
                },
                curfewNoNew: false,
                dayPnlNoNew: false,
                allowedNow: new[] { "S2", "S3", "S6", "S11" },
                learnerOn: true,
                learnerLastRun: DateTime.Now.AddMinutes(-15),
                learnerApplied: true,
                learnerNote: "Model updated successfully",
                strategyPnl: new Dictionary<string, object>
                {
                    ["S2"] = new { trades = 5, pnl = 125.50m, winRate = 80.0 },
                    ["S3"] = new { trades = 3, pnl = -45.75m, winRate = 33.3 },
                    ["S6"] = new { trades = 2, pnl = 85.25m, winRate = 100.0 },
                    ["S11"] = new { trades = 4, pnl = -85.25m, winRate = 50.0 }
                },
                healthStatus: "HEALTHY",
                healthDetails: new Dictionary<string, object>
                {
                    ["mlPersistence"] = new { status = "Healthy", message = "All models saved successfully" },
                    ["strategyConfig"] = new { status = "Healthy", message = "All strategies configured correctly" },
                    ["dataFeeds"] = new { status = "Healthy", message = "Real-time data flowing" },
                    ["riskManagement"] = new { status = "Healthy", message = "Risk controls operating normally" }
                },
                selfHealingStatus: new
                {
                    availableActions = 8,
                    attemptsToday = 2,
                    successfulToday = 2,
                    successRate = 100.0
                }
            );
        }
    }
}