using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.Http;
using System.Text.Json;
using System.Collections.Concurrent;

namespace StandaloneDashboard
{
    public class Program
    {
        private static readonly ConcurrentQueue<string> _logEntries = new();
        private static readonly Random _random = new();

        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);
            var app = builder.Build();

            // Configure the HTTP request pipeline
            app.UseDefaultFiles();
            app.UseStaticFiles();

            // Root redirect
            app.MapGet("/", () => Results.Redirect("/dashboard"));

            // Serve unified dashboard
            app.MapGet("/dashboard", async ctx =>
            {
                ctx.Response.ContentType = "text/html; charset=utf-8";
                await ctx.Response.SendFileAsync(Path.Combine(app.Environment.ContentRootPath, "wwwroot", "unified-dashboard.html"));
            });

            // Health endpoint
            app.MapGet("/healthz", () => Results.Json(new { 
                ok = true, 
                msg = "Unified Dashboard Running", 
                mode = "DEMO",
                timestamp = DateTime.UtcNow 
            }));

            // Bot control endpoints
            app.MapPost("/api/bot/start", async (HttpContext ctx) =>
            {
                try
                {
                    var body = await ctx.Request.ReadFromJsonAsync<BotControlRequest>();
                    var mode = body?.Mode ?? "paper";
                    Console.WriteLine($"[Demo] Bot start requested - Mode: {mode}");
                    AddLog("info", $"🚀 Bot started in {mode.ToUpper()} mode (Demo)");
                    return Results.Json(new { success = true, message = $"Bot started in {mode.ToUpper()} mode" });
                }
                catch (Exception ex)
                {
                    AddLog("error", $"Failed to start bot: {ex.Message}");
                    return Results.Json(new { success = false, error = ex.Message });
                }
            });

            app.MapPost("/api/bot/stop", () =>
            {
                Console.WriteLine("[Demo] Bot stop requested");
                AddLog("info", "⏹️ Bot stopped (Demo)");
                return Results.Json(new { success = true, message = "Bot stopped" });
            });

            app.MapPost("/api/bot/mode", async (HttpContext ctx) =>
            {
                try
                {
                    var body = await ctx.Request.ReadFromJsonAsync<BotControlRequest>();
                    var mode = body?.Mode ?? "paper";
                    AddLog("info", $"Mode changed to {mode.ToUpper()}");
                    return Results.Json(new { success = true, message = $"Mode changed to {mode.ToUpper()}" });
                }
                catch (Exception ex)
                {
                    return Results.Json(new { success = false, error = ex.Message });
                }
            });

            // Real-time data stream (Server-Sent Events)
            app.MapGet("/stream/realtime", async (HttpContext ctx) =>
            {
                ctx.Response.Headers.Add("Content-Type", "text/event-stream");
                ctx.Response.Headers.Add("Cache-Control", "no-cache");
                ctx.Response.Headers.Add("Connection", "keep-alive");
                ctx.Response.Headers.Add("Access-Control-Allow-Origin", "*");

                try
                {
                    while (!ctx.RequestAborted.IsCancellationRequested)
                    {
                        // Send demo data
                        var demoData = new
                        {
                            overview = new
                            {
                                accountBalance = 100000 + (_random.NextDouble() - 0.5) * 1000,
                                totalPnL = (_random.NextDouble() - 0.3) * 5000,
                                openPositions = _random.Next(0, 4),
                                todayTrades = _random.Next(5, 30),
                                botMode = "DEMO",
                                activeStrategy = "Multiple"
                            },
                            learning = new
                            {
                                accuracy = 85.0 + _random.NextDouble() * 10,
                                samples = 15000 + _random.Next(100, 500),
                                uptime = 94.0 + _random.NextDouble() * 5,
                                lastTraining = $"{_random.Next(1, 4)}h ago"
                            },
                            timestamp = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                        };

                        var json = JsonSerializer.Serialize(demoData);
                        await ctx.Response.WriteAsync($"data: {json}\n\n");
                        await ctx.Response.Body.FlushAsync();

                        await Task.Delay(5000, ctx.RequestAborted); // Update every 5 seconds
                    }
                }
                catch (OperationCanceledException)
                {
                    // Client disconnected
                }
            });

            // Historical data endpoint (demo)
            app.MapGet("/data/history", (string symbol, string res, long from, long to) =>
            {
                var bars = GenerateDemoHistoricalData(symbol, from, to);
                return Results.Json(bars);
            });

            // Demo data simulation background task
            _ = Task.Run(async () =>
            {
                await Task.Delay(2000); // Wait for startup
                AddLog("success", "🚀 Unified Dashboard initialized successfully");
                AddLog("info", "📊 All bot features available in demo mode");
                AddLog("learning", "🧠 Cloud learning simulation active");
                
                while (true)
                {
                    try
                    {
                        // Simulate periodic activities
                        if (_random.NextDouble() > 0.8)
                        {
                            var symbols = new[] { "ES", "NQ" };
                            var sides = new[] { "BUY", "SELL" };
                            var symbol = symbols[_random.Next(symbols.Length)];
                            var side = sides[_random.Next(sides.Length)];
                            var price = symbol == "ES" ? 4500 + _random.NextDouble() * 100 : 15000 + _random.NextDouble() * 500;
                            var qty = _random.Next(1, 4);
                            
                            AddLog("info", $"📈 [Demo] {symbol} {side} {qty} @ {price:F2}");
                        }

                        if (_random.NextDouble() > 0.9)
                        {
                            AddLog("learning", "🤖 Cloud learning cycle completed");
                        }

                        await Task.Delay(10000); // Log activity every 10 seconds
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Demo simulation error: {ex.Message}");
                        await Task.Delay(30000);
                    }
                }
            });

            Console.WriteLine("🚀 Unified Cloud Dashboard Starting...");
            Console.WriteLine($"📊 Dashboard: http://localhost:5050/dashboard");
            Console.WriteLine($"💾 Health: http://localhost:5050/healthz");
            Console.WriteLine($"🔄 Real-time: http://localhost:5050/stream/realtime");
            Console.WriteLine("🧪 Running in Demo Mode - All bot features available");
            Console.WriteLine("⚡ Features: Bot Control, Live Data, Cloud Learning, Health Monitoring");

            app.Run("http://localhost:5050");
        }

        private static void AddLog(string type, string message)
        {
            var timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            var logEntry = $"[{timestamp}] {message}";
            _logEntries.Enqueue(logEntry);
            
            // Keep only last 100 entries
            while (_logEntries.Count > 100)
            {
                _logEntries.TryDequeue(out _);
            }
            
            Console.WriteLine($"[{type.ToUpper()}] {logEntry}");
        }

        private static object[] GenerateDemoHistoricalData(string symbol, long from, long to)
        {
            var bars = new List<object>();
            var startPrice = symbol == "ES" ? 4500.0 : 15000.0;
            var current = from;
            var price = startPrice;

            while (current <= to && bars.Count < 1000)
            {
                price += (_random.NextDouble() - 0.5) * 10;
                price = Math.Max(price, startPrice * 0.9); // Floor
                price = Math.Min(price, startPrice * 1.1); // Ceiling

                bars.Add(new
                {
                    time = current,
                    open = price,
                    high = price + _random.NextDouble() * 5,
                    low = price - _random.NextDouble() * 5,
                    close = price + (_random.NextDouble() - 0.5) * 3,
                    volume = _random.Next(1000, 10000)
                });

                current += 60; // 1-minute bars
            }

            return bars.ToArray();
        }
    }

    public class BotControlRequest
    {
        public string? Mode { get; set; }
        public bool CloudLearning { get; set; }
    }
}