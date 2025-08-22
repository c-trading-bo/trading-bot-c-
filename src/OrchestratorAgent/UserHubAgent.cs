using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;

namespace YourNamespace
{
    public class UserHubAgent
    {
        private readonly HubConnection _hub;
        private readonly ILogger<UserHubAgent> _log;

        public UserHubAgent(ILogger<UserHubAgent> log)
        {
            _log = log;
        }

        public Microsoft.AspNetCore.SignalR.Client.HubConnection? Connection => _hub;

        public async Task StartAsync()
        {
            await _hub.StartAsync();
        }

        public async Task StopAsync()
        {
            await _hub.StopAsync();
        }

        // Add your existing methods and properties here
    }

    // Minimal stub for SignalRLoggerProvider to fix build error
    public class SignalRLoggerProvider : Microsoft.Extensions.Logging.ILoggerProvider
    {
        private readonly Microsoft.Extensions.Logging.ILogger _logger;
        public SignalRLoggerProvider(Microsoft.Extensions.Logging.ILogger logger)
        {
            _logger = logger;
        }
        public Microsoft.Extensions.Logging.ILogger CreateLogger(string categoryName) => _logger;
        public void Dispose() { }
    }
}