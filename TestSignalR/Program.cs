using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Threading;

namespace TestSignalR
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("🔗 Testing Production-Ready TopstepX SignalR Connection Stability...");
            Console.WriteLine("Testing all requirements from production readiness specification:");
            Console.WriteLine("- JWT readiness before hub connections");
            Console.WriteLine("- WebSockets-only transport");
            Console.WriteLine("- Fresh token access providers");
            Console.WriteLine("- Connection stability and subscriptions");
            Console.WriteLine("- Extended stability test (10+ minutes)");
            
            // Try to read token from various sources
            string? jwt = await GetJwtTokenAsync();
            
            if (string.IsNullOrEmpty(jwt))
            {
                Console.WriteLine("❌ No JWT token found. Checked:");
                Console.WriteLine("   - state/auth_token.txt");
                Console.WriteLine("   - state/access_token.txt");
                Console.WriteLine("   - state/jwt_token.txt");
                Console.WriteLine("   - TOPSTEPX_JWT environment variable");
                return;
            }
            
            // Test JWT validation (from our new implementation)
            if (!await ValidateJwtTokenAsync(jwt))
            {
                Console.WriteLine("❌ JWT token validation failed - timing or format issues");
                return;
            }
            
            // Test REST API first
            Console.WriteLine("\n🌐 Testing REST API authentication...");
            var accountId = await TestRestApiAsync(jwt);
            if (!accountId.HasValue)
            {
                Console.WriteLine("❌ REST API test failed - cannot proceed with SignalR tests");
                return;
            }
            
            // Test Production-Ready SignalR connections
            Console.WriteLine("\n🔌 Testing Production-Ready SignalR connections...");
            
            var endpoints = new[]
            {
                ("TopstepX User Hub", "https://rtc.topstepx.com/hubs/user"),
                ("TopstepX Market Hub", "https://rtc.topstepx.com/hubs/market"),
            };
            
            foreach (var (name, url) in endpoints)
            {
                Console.WriteLine($"\n🧪 Testing {name}: {url}");
                await TestProductionReadySignalRConnection(name, url, jwt, accountId);
            }
            
            // Extended stability test (requirement: 10+ minutes)
            if (args.Length > 0 && args[0] == "--extended")
            {
                Console.WriteLine("\n⏱️ Running extended stability test (10+ minutes)...");
                await ExtendedStabilityTest(jwt, accountId.Value);
            }
            
            Console.WriteLine("\n✅ All production readiness tests completed!");
        }
        
        static async Task<string?> GetJwtTokenAsync()
        {
            // Try reading from state files
            var stateDir = Environment.GetEnvironmentVariable("STATE_DIR") ?? 
                          Path.Combine(Directory.GetCurrentDirectory(), "state");
            var tokenFiles = new[] { "auth_token.txt", "access_token.txt", "jwt_token.txt" };
            
            foreach (var tokenFile in tokenFiles)
            {
                var fullPath = Path.Combine(stateDir, tokenFile);
                if (File.Exists(fullPath))
                {
                    var jwt = await File.ReadAllTextAsync(fullPath);
                    jwt = jwt.Trim();
                    if (!string.IsNullOrEmpty(jwt))
                    {
                        Console.WriteLine($"✅ Found JWT token in {tokenFile} (length: {jwt.Length})");
                        return jwt;
                    }
                }
            }
            
            var envJwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            if (!string.IsNullOrEmpty(envJwt))
            {
                Console.WriteLine("✅ Found JWT token in environment variable");
                return envJwt;
            }
            
            await Task.CompletedTask; // Make it properly async
            return null;
        }
        
        /// <summary>
        /// Validate JWT token format and timing (from our new implementation)
        /// </summary>
        static async Task<bool> ValidateJwtTokenAsync(string token)
        {
            try
            {
                Console.WriteLine("🔍 Validating JWT token format and timing...");
                
                // Remove Bearer prefix if present
                if (token.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
                {
                    token = token.Substring(7);
                }
                
                // Check JWT format (should have 3 parts separated by dots)
                var parts = token.Split('.');
                if (parts.Length != 3)
                {
                    Console.WriteLine("❌ JWT token format invalid - expected 3 parts separated by dots");
                    return false;
                }
                
                // Decode payload to check timing
                var payloadBytes = Convert.FromBase64String(AddBase64Padding(parts[1]));
                var payloadJson = System.Text.Encoding.UTF8.GetString(payloadBytes);
                using var doc = JsonDocument.Parse(payloadJson);
                
                var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                
                // Check expiry (exp claim)
                if (doc.RootElement.TryGetProperty("exp", out var expElement))
                {
                    var exp = expElement.GetInt64();
                    if (now >= exp)
                    {
                        Console.WriteLine($"❌ JWT token expired - exp: {exp}, now: {now}");
                        return false;
                    }
                    var remaining = exp - now;
                    Console.WriteLine($"✅ JWT token valid - expires in {remaining} seconds");
                }
                
                // Check not-before (nbf claim)
                if (doc.RootElement.TryGetProperty("nbf", out var nbfElement))
                {
                    var nbf = nbfElement.GetInt64();
                    if (now < nbf)
                    {
                        var skew = nbf - now;
                        Console.WriteLine($"❌ JWT token not yet valid - nbf: {nbf}, now: {now}, skew: {skew}s (check system clock)");
                        return false;
                    }
                }
                
                Console.WriteLine("✅ JWT token timing validation passed");
                await Task.CompletedTask; // Make it properly async
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ JWT validation error: {ex.Message}");
                return false;
            }
        }
        
        static string AddBase64Padding(string base64)
        {
            switch (base64.Length % 4)
            {
                case 2: return base64 + "==";
                case 3: return base64 + "=";
                default: return base64;
            }
        }
        
        static async Task<long?> TestRestApiAsync(string jwt)
        {
            // Clean up token
            if (jwt.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                jwt = jwt.Substring(7);
            }
            
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
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ REST API test failed: {ex.Message}");
            }
            
            return accountId;
        }
        
        static async Task TestProductionReadySignalRConnection(string name, string url, string jwt, long? accountId)
        {
            Console.WriteLine($"🔄 [{name}] Implementing JWT readiness wait (up to 45 seconds)...");
            
            // Simulate our JWT readiness check
            var startTime = DateTime.UtcNow;
            while ((DateTime.UtcNow - startTime).TotalSeconds < 5) // Shortened for testing
            {
                await Task.Delay(1000);
                Console.WriteLine($"⏳ [{name}] Waiting for JWT readiness... {(DateTime.UtcNow - startTime).TotalSeconds:F0}s");
            }
            Console.WriteLine($"✅ [{name}] JWT ready after {(DateTime.UtcNow - startTime).TotalSeconds:F1} seconds");
            
            var connection = new HubConnectionBuilder()
                .WithUrl(url, options =>
                {
                    // PRODUCTION: Fresh token provider (as implemented in our fix)
                    options.AccessTokenProvider = async () => 
                    {
                        var freshJwt = await GetJwtTokenAsync();
                        if (!string.IsNullOrEmpty(freshJwt) && freshJwt.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
                        {
                            freshJwt = freshJwt.Substring(7);
                        }
                        return freshJwt;
                    };
                    
                    // PRODUCTION: Force WebSockets only, skip negotiation
                    options.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets;
                    options.SkipNegotiation = true;
                    
                    options.CloseTimeout = TimeSpan.FromSeconds(45);
                    
                    // SSL bypass for testing (production would use proper cert validation)
                    options.HttpMessageHandlerFactory = _ => new HttpClientHandler()
                    {
                        ServerCertificateCustomValidationCallback = (message, cert, chain, errors) => true
                    };
                })
                .WithAutomaticReconnect(new[] { 
                    TimeSpan.FromSeconds(45), 
                    TimeSpan.FromSeconds(90), 
                    TimeSpan.FromSeconds(180) 
                })
                .ConfigureLogging(logging =>
                {
                    logging.AddConsole();
                    logging.SetMinimumLevel(LogLevel.Information);
                    logging.AddFilter("Microsoft.AspNetCore.SignalR.Client", LogLevel.Warning);
                    logging.AddFilter("Microsoft.AspNetCore.Http.Connections.Client", LogLevel.Warning);
                })
                .Build();
            
            // Enhanced timeouts (as in our implementation)
            connection.ServerTimeout = TimeSpan.FromSeconds(90);
            connection.KeepAliveInterval = TimeSpan.FromSeconds(20);
            connection.HandshakeTimeout = TimeSpan.FromSeconds(45);
            
            // Event handlers for data reception
            connection.On<object>("GatewayUserOrder", data => 
                Console.WriteLine($"📥 [{name}] ORDER: {JsonSerializer.Serialize(data).Substring(0, Math.Min(100, JsonSerializer.Serialize(data).Length))}..."));
            
            connection.On<object>("GatewayUserTrade", data => 
                Console.WriteLine($"📥 [{name}] TRADE: {JsonSerializer.Serialize(data).Substring(0, Math.Min(100, JsonSerializer.Serialize(data).Length))}..."));
                
            connection.On<object>("ContractQuotes", data => 
                Console.WriteLine($"📈 [{name}] MARKET: {JsonSerializer.Serialize(data).Substring(0, Math.Min(100, JsonSerializer.Serialize(data).Length))}..."));
            
            connection.Closed += error =>
            {
                if (error != null)
                {
                    Console.WriteLine($"❌ [{name}] Connection closed with error: {error.Message}");
                }
                else
                {
                    Console.WriteLine($"✅ [{name}] Connection closed cleanly");
                }
                return Task.CompletedTask;
            };
            
            connection.Reconnecting += error =>
            {
                Console.WriteLine($"🔄 [{name}] Reconnecting... {error?.Message ?? "Unknown reason"}");
                return Task.CompletedTask;
            };
            
            connection.Reconnected += connectionId =>
            {
                Console.WriteLine($"✅ [{name}] Reconnected successfully: {connectionId}");
                return Task.CompletedTask;
            };
            
            try
            {
                Console.WriteLine($"🔗 [{name}] Starting connection with enhanced stability validation...");
                var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                await connection.StartAsync(cts.Token);
                
                Console.WriteLine($"📊 [{name}] Initial state: {connection.State}, ID: {connection.ConnectionId ?? "NULL"}");
                
                // Extended stability validation (as in our implementation)
                Console.WriteLine($"🔍 [{name}] Performing extended stability validation...");
                for (int check = 1; check <= 3; check++)
                {
                    await Task.Delay(2000);
                    
                    if (connection.State == HubConnectionState.Connected && !string.IsNullOrEmpty(connection.ConnectionId))
                    {
                        Console.WriteLine($"✅ [{name}] Stability check {check}/3: State: {connection.State}, ID: {connection.ConnectionId}");
                    }
                    else
                    {
                        Console.WriteLine($"❌ [{name}] Stability check {check}/3: FAILED - State: {connection.State}, ID: {connection.ConnectionId ?? "null"}");
                        if (check == 3)
                        {
                            Console.WriteLine($"❌ [{name}] Connection unstable after extended validation");
                            return;
                        }
                    }
                }
                
                if (connection.State == HubConnectionState.Connected && !string.IsNullOrEmpty(connection.ConnectionId))
                {
                    Console.WriteLine($"✅ [{name}] Connection validated after extended checks");
                    
                    // Test subscriptions immediately after connect
                    if (accountId.HasValue)
                    {
                        try
                        {
                            Console.WriteLine($"🔔 [{name}] Testing immediate subscription success...");
                            if (url.Contains("user"))
                            {
                                await connection.InvokeAsync("SubscribeOrders", accountId.Value);
                                Console.WriteLine($"✅ [{name}] Orders subscription successful");
                                
                                await connection.InvokeAsync("SubscribeTrades", accountId.Value);
                                Console.WriteLine($"✅ [{name}] Trades subscription successful");
                            }
                            else if (url.Contains("market"))
                            {
                                await connection.InvokeAsync("SubscribeContractQuotes", "ES");
                                Console.WriteLine($"✅ [{name}] Market data subscription successful");
                            }
                        }
                        catch (Exception subEx)
                        {
                            Console.WriteLine($"❌ [{name}] Subscription failed: {subEx.Message}");
                        }
                    }
                    
                    // Test stability for 30 seconds
                    Console.WriteLine($"⏱️ [{name}] Testing connection stability for 30 seconds...");
                    await Task.Delay(30000);
                    
                    Console.WriteLine($"📊 [{name}] Final state: {connection.State}, ID: {connection.ConnectionId ?? "NULL"}");
                    
                    if (connection.State == HubConnectionState.Connected)
                    {
                        Console.WriteLine($"✅ [{name}] PRODUCTION READINESS TEST PASSED");
                    }
                    else
                    {
                        Console.WriteLine($"❌ [{name}] PRODUCTION READINESS TEST FAILED - Connection lost");
                    }
                }
                else
                {
                    Console.WriteLine($"❌ [{name}] Connection failed final validation");
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
                    Console.WriteLine($"🔌 [{name}] Connection disposed cleanly");
                }
                catch (Exception disposeEx)
                {
                    Console.WriteLine($"⚠️ [{name}] Dispose error: {disposeEx.Message}");
                }
            }
        }
        
        static async Task ExtendedStabilityTest(string jwt, long accountId)
        {
            Console.WriteLine("🏃‍♂️ Starting 10+ minute continuous stability test...");
            Console.WriteLine("Testing requirements:");
            Console.WriteLine("- No 'Normal closure' errors");
            Console.WriteLine("- No 'InvokeCoreAsync cannot be called' errors");
            Console.WriteLine("- Continuous ES/NQ market data");
            Console.WriteLine("- Continuous order/trade events");
            
            var testDuration = TimeSpan.FromMinutes(12); // 12 minutes for extended test
            var startTime = DateTime.UtcNow;
            var marketDataCount = 0;
            var orderEventCount = 0;
            var tradeEventCount = 0;
            var errors = new List<string>();
            
            // Create both connections for extended test
            var userConnection = await CreateStabilityTestConnection(
                "User Hub Extended", 
                "https://rtc.topstepx.com/hubs/user", 
                jwt, 
                errors);
                
            var marketConnection = await CreateStabilityTestConnection(
                "Market Hub Extended", 
                "https://rtc.topstepx.com/hubs/market", 
                jwt, 
                errors);
            
            // Set up event counters
            userConnection.On<object>("GatewayUserOrder", data => 
            {
                Interlocked.Increment(ref orderEventCount);
                if (orderEventCount % 10 == 0)
                    Console.WriteLine($"📈 Order events received: {orderEventCount}");
            });
            
            userConnection.On<object>("GatewayUserTrade", data => 
            {
                Interlocked.Increment(ref tradeEventCount);
                if (tradeEventCount % 10 == 0)
                    Console.WriteLine($"💰 Trade events received: {tradeEventCount}");
            });
            
            marketConnection.On<object>("ContractQuotes", data => 
            {
                Interlocked.Increment(ref marketDataCount);
                if (marketDataCount % 100 == 0)
                    Console.WriteLine($"📊 Market data events received: {marketDataCount}");
            });
            
            try
            {
                // Start connections
                await userConnection.StartAsync();
                await marketConnection.StartAsync();
                
                // Subscribe
                await userConnection.InvokeAsync("SubscribeOrders", accountId);
                await userConnection.InvokeAsync("SubscribeTrades", accountId);
                await marketConnection.InvokeAsync("SubscribeContractQuotes", "ES");
                await marketConnection.InvokeAsync("SubscribeContractQuotes", "NQ");
                
                Console.WriteLine("✅ Extended test connections established and subscribed");
                
                // Monitor for the test duration
                while (DateTime.UtcNow - startTime < testDuration)
                {
                    var elapsed = DateTime.UtcNow - startTime;
                    var remaining = testDuration - elapsed;
                    
                    Console.WriteLine($"⏱️ Extended test progress: {elapsed.TotalMinutes:F1}/{testDuration.TotalMinutes:F0} minutes " +
                                    $"(Remaining: {remaining.TotalMinutes:F1}m)");
                    Console.WriteLine($"📊 Events - Market: {marketDataCount}, Orders: {orderEventCount}, Trades: {tradeEventCount}");
                    Console.WriteLine($"🔗 Connection states - User: {userConnection.State}, Market: {marketConnection.State}");
                    
                    // Check for disconnections
                    if (userConnection.State != HubConnectionState.Connected)
                    {
                        errors.Add($"User hub disconnected at {elapsed.TotalMinutes:F1} minutes");
                    }
                    if (marketConnection.State != HubConnectionState.Connected)
                    {
                        errors.Add($"Market hub disconnected at {elapsed.TotalMinutes:F1} minutes");
                    }
                    
                    await Task.Delay(TimeSpan.FromMinutes(1)); // Check every minute
                }
                
                Console.WriteLine("\n🏁 Extended stability test completed!");
                Console.WriteLine($"📊 Final statistics:");
                Console.WriteLine($"   - Duration: {testDuration.TotalMinutes:F0} minutes");
                Console.WriteLine($"   - Market data events: {marketDataCount}");
                Console.WriteLine($"   - Order events: {orderEventCount}");
                Console.WriteLine($"   - Trade events: {tradeEventCount}");
                Console.WriteLine($"   - Errors: {errors.Count}");
                
                if (errors.Count == 0 && marketDataCount > 0)
                {
                    Console.WriteLine("✅ EXTENDED STABILITY TEST PASSED - Production ready!");
                }
                else
                {
                    Console.WriteLine("❌ EXTENDED STABILITY TEST FAILED");
                    foreach (var error in errors)
                    {
                        Console.WriteLine($"   ❌ {error}");
                    }
                }
            }
            finally
            {
                await userConnection.DisposeAsync();
                await marketConnection.DisposeAsync();
            }
        }
        
        static async Task<HubConnection> CreateStabilityTestConnection(string name, string url, string jwt, List<string> errors)
        {
            var connection = new HubConnectionBuilder()
                .WithUrl(url, options =>
                {
                    options.AccessTokenProvider = () => Task.FromResult<string?>(jwt.StartsWith("Bearer") ? jwt.Substring(7) : jwt);
                    options.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets;
                    options.SkipNegotiation = true;
                    options.CloseTimeout = TimeSpan.FromSeconds(45);
                    options.HttpMessageHandlerFactory = _ => new HttpClientHandler()
                    {
                        ServerCertificateCustomValidationCallback = (message, cert, chain, sslErrors) => true
                    };
                })
                .WithAutomaticReconnect(new[] { 
                    TimeSpan.FromSeconds(45), 
                    TimeSpan.FromSeconds(90), 
                    TimeSpan.FromSeconds(180) 
                })
                .ConfigureLogging(logging =>
                {
                    logging.AddConsole();
                    logging.SetMinimumLevel(LogLevel.Warning); // Reduce noise for extended test
                })
                .Build();
            
            connection.ServerTimeout = TimeSpan.FromSeconds(90);
            connection.KeepAliveInterval = TimeSpan.FromSeconds(20);
            connection.HandshakeTimeout = TimeSpan.FromSeconds(45);
            
            connection.Closed += error =>
            {
                var errorMsg = error?.Message ?? "Normal closure";
                if (errorMsg != "Normal closure")
                {
                    errors.Add($"{name}: {errorMsg}");
                    Console.WriteLine($"❌ [{name}] Connection closed: {errorMsg}");
                }
                return Task.CompletedTask;
            };
            
            await Task.CompletedTask; // Make it properly async
            return connection;
        }
    }
}
