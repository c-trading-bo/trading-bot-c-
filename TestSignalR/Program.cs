using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace TestSignalR
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("🔗 Testing TopstepX SignalR Connection...");
            
            // Try to read token from various sources
            string? jwt = null;
            
            // Try reading from state files
            var stateDir = @"C:\Users\kevin\Downloads\C# ai bot\trading-bot-c--1\state";
            var tokenFiles = new[] { "auth_token.txt", "access_token.txt", "jwt_token.txt" };
            
            foreach (var tokenFile in tokenFiles)
            {
                var fullPath = Path.Combine(stateDir, tokenFile);
                if (File.Exists(fullPath))
                {
                    jwt = File.ReadAllText(fullPath).Trim();
                    if (!string.IsNullOrEmpty(jwt))
                    {
                        Console.WriteLine($"✅ Found JWT token in {tokenFile} (length: {jwt.Length})");
                        break;
                    }
                }
            }
            
            if (string.IsNullOrEmpty(jwt))
            {
                jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                if (!string.IsNullOrEmpty(jwt))
                {
                    Console.WriteLine("✅ Found JWT token in environment variable");
                }
            }
            
            if (string.IsNullOrEmpty(jwt))
            {
                Console.WriteLine("❌ No JWT token found. Checked:");
                Console.WriteLine("   - state/auth_token.txt");
                Console.WriteLine("   - state/access_token.txt");
                Console.WriteLine("   - state/jwt_token.txt");
                Console.WriteLine("   - TOPSTEPX_JWT environment variable");
                return;
            }
            
            // Clean up token
            if (jwt.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                jwt = jwt.Substring(7);
            }
            
            // Test REST API first
            Console.WriteLine("\n🌐 Testing REST API authentication...");
            using var httpClient = new HttpClient();
            httpClient.DefaultRequestHeaders.Authorization = 
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
            
            long? accountId = null;
            try
            {
                var response = await httpClient.GetAsync("https://api.topstepx.com/api/Account");
                Console.WriteLine($"REST API Status: {response.StatusCode}");
                
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"✅ Account data received (length: {content.Length})");
                    
                    // Parse account ID
                    try
                    {
                        var accountData = JsonSerializer.Deserialize<JsonElement>(content);
                        if (accountData.TryGetProperty("data", out var dataArray) && dataArray.ValueKind == JsonValueKind.Array)
                        {
                            foreach (var account in dataArray.EnumerateArray())
                            {
                                if (account.TryGetProperty("accountId", out var accountIdProp))
                                {
                                    accountId = accountIdProp.GetInt64();
                                    Console.WriteLine($"✅ Found account ID: {accountId}");
                                    break;
                                }
                            }
                        }
                    }
                    catch (Exception parseEx)
                    {
                        Console.WriteLine($"⚠️ Failed to parse account data: {parseEx.Message}");
                    }
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    Console.WriteLine($"❌ REST API Error: {errorContent}");
                    return;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ REST API test failed: {ex.Message}");
                return;
            }
            
            // Test SignalR connections
            Console.WriteLine("\n🔌 Testing SignalR connections...");
            
            var endpoints = new[]
            {
                ("TopstepX Original", "https://rtc.topstepx.com/hubs/user"),
                ("ProjectX Gateway", "https://gateway-rtc-demo.s2f.projectx.com/hubs/user")
            };
            
            foreach (var (name, url) in endpoints)
            {
                Console.WriteLine($"\n🧪 Testing {name}: {url}");
                await TestSignalRConnection(name, url, jwt, accountId);
            }
            
            Console.WriteLine("\n✅ All tests completed!");
        }
        
        static async Task TestSignalRConnection(string name, string url, string jwt, long? accountId)
        {
            var connection = new HubConnectionBuilder()
                .WithUrl(url, options =>
                {
                    options.AccessTokenProvider = () => Task.FromResult<string?>(jwt);
                    options.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets;
                    options.SkipNegotiation = true;
                    
                    // SSL bypass for testing
                    options.HttpMessageHandlerFactory = _ => new HttpClientHandler()
                    {
                        ServerCertificateCustomValidationCallback = (message, cert, chain, errors) => true
                    };
                })
                .WithAutomaticReconnect(new[] { TimeSpan.Zero, TimeSpan.FromSeconds(2) })
                .ConfigureLogging(logging =>
                {
                    logging.AddConsole();
                    logging.SetMinimumLevel(LogLevel.Information);
                })
                .Build();
            
            // Event handlers
            connection.On<object>("GatewayUserOrder", data => 
                Console.WriteLine($"📥 [{name}] ORDER: {data}"));
            
            connection.On<object>("GatewayUserTrade", data => 
                Console.WriteLine($"📥 [{name}] TRADE: {data}"));
            
            connection.Closed += error =>
            {
                Console.WriteLine($"🔌 [{name}] Connection closed: {error?.Message ?? "Clean"}");
                return Task.CompletedTask;
            };
            
            try
            {
                Console.WriteLine($"🔗 [{name}] Starting connection...");
                await connection.StartAsync();
                
                Console.WriteLine($"📊 [{name}] State: {connection.State}, ID: {connection.ConnectionId ?? "NULL"}");
                
                if (connection.State == HubConnectionState.Connected && !string.IsNullOrEmpty(connection.ConnectionId))
                {
                    Console.WriteLine($"✅ [{name}] Connected successfully!");
                    
                    // Stabilization delay
                    await Task.Delay(1000);
                    Console.WriteLine($"📊 [{name}] After delay - State: {connection.State}, ID: {connection.ConnectionId ?? "NULL"}");
                    
                    if (accountId.HasValue && connection.State == HubConnectionState.Connected)
                    {
                        try
                        {
                            Console.WriteLine($"🔔 [{name}] Subscribing to events for account {accountId}...");
                            await connection.InvokeAsync("SubscribeOrders", accountId.Value);
                            Console.WriteLine($"✅ [{name}] Orders subscription successful");
                            
                            await connection.InvokeAsync("SubscribeTrades", accountId.Value);
                            Console.WriteLine($"✅ [{name}] Trades subscription successful");
                        }
                        catch (Exception subEx)
                        {
                            Console.WriteLine($"❌ [{name}] Subscription failed: {subEx.Message}");
                        }
                    }
                    
                    // Keep alive for 10 seconds
                    Console.WriteLine($"⏱️ [{name}] Testing stability for 10 seconds...");
                    await Task.Delay(10000);
                    
                    Console.WriteLine($"📊 [{name}] Final state - State: {connection.State}, ID: {connection.ConnectionId ?? "NULL"}");
                }
                else
                {
                    Console.WriteLine($"❌ [{name}] Connection failed or no ConnectionId");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ [{name}] Connection exception: {ex.Message}");
            }
            finally
            {
                try
                {
                    await connection.DisposeAsync();
                    Console.WriteLine($"🔌 [{name}] Connection disposed");
                }
                catch (Exception disposeEx)
                {
                    Console.WriteLine($"⚠️ [{name}] Dispose error: {disposeEx.Message}");
                }
            }
        }
    }
}
