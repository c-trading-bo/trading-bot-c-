using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;

// Quick SignalR connection test
class Program
{
    static async Task Main(string[] args)
    {
        var logger = LoggerFactory.Create(builder => builder.AddConsole())
            .CreateLogger<Program>();

        // Get JWT token from environment
        var jwt = Environment.GetEnvironmentVariable("TOPSTEPX_AUTH_TOKEN");
        if (string.IsNullOrEmpty(jwt))
        {
            Console.WriteLine("❌ No JWT token found in TOPSTEPX_AUTH_TOKEN");
            return;
        }

        Console.WriteLine($"✅ Found JWT token: {jwt.Substring(0, Math.Min(20, jwt.Length))}...");

        // Remove Bearer prefix if present
        if (jwt.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
        {
            jwt = jwt.Substring(7);
            Console.WriteLine("🔧 Removed Bearer prefix");
        }

        try
        {
            Console.WriteLine("🔌 Testing User Hub connection...");
            var userHub = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/user", options =>
                {
                    options.AccessTokenProvider = () => Task.FromResult<string?>(jwt);
                })
                .WithAutomaticReconnect()
                .Build();

            // Set timeouts
            userHub.ServerTimeout = TimeSpan.FromSeconds(60);
            userHub.KeepAliveInterval = TimeSpan.FromSeconds(15);
            userHub.HandshakeTimeout = TimeSpan.FromSeconds(30);

            Console.WriteLine($"🔄 Starting User Hub connection (initial state: {userHub.State})...");
            await userHub.StartAsync(CancellationToken.None);
            Console.WriteLine($"✅ User Hub connected! State: {userHub.State}, ConnectionId: {userHub.ConnectionId}");

            Console.WriteLine("🔌 Testing Market Hub connection...");
            var marketHub = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/market", options =>
                {
                    options.AccessTokenProvider = () => Task.FromResult<string?>(jwt);
                })
                .WithAutomaticReconnect()
                .Build();

            // Set timeouts
            marketHub.ServerTimeout = TimeSpan.FromSeconds(60);
            marketHub.KeepAliveInterval = TimeSpan.FromSeconds(15);
            marketHub.HandshakeTimeout = TimeSpan.FromSeconds(30);

            Console.WriteLine($"🔄 Starting Market Hub connection (initial state: {marketHub.State})...");
            await marketHub.StartAsync(CancellationToken.None);
            Console.WriteLine($"✅ Market Hub connected! State: {marketHub.State}, ConnectionId: {marketHub.ConnectionId}");

            Console.WriteLine("🎉 Both SignalR connections established successfully!");

            // Keep alive for a moment
            await Task.Delay(2000);

            await userHub.DisposeAsync();
            await marketHub.DisposeAsync();
            Console.WriteLine("🛑 Connections disposed");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ SignalR connection failed: {ex.Message}");
            Console.WriteLine($"Exception type: {ex.GetType().Name}");
            if (ex.InnerException != null)
            {
                Console.WriteLine($"Inner exception: {ex.InnerException.Message}");
            }
        }
    }
}
