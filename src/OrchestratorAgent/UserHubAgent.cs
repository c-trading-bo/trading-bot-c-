using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using SupervisorAgent;

namespace YourNamespace
{
    public sealed class UserHubAgent
    {
        private readonly ILogger<UserHubAgent> _log;
        private readonly StatusService _statusService;
        private HubConnection _hub = default!;   // fixes CS0649
        private bool _wired;                     // ensure handlers wired once
        private readonly SemaphoreSlim _gate = new(1, 1);

        private void LogTokenExp(string jwt)
        {
            try
            {
                var parts = jwt.Split('.');
                if (parts.Length == 3)
                {
                    var payloadJson = System.Text.Encoding.UTF8.GetString(Convert.FromBase64String(PadB64(parts[1])));
                    using var doc = System.Text.Json.JsonDocument.Parse(payloadJson);
                    if (doc.RootElement.TryGetProperty("exp", out var expEl))
                    {
                        var expUnix = expEl.GetInt64();
                        var expUtc = DateTimeOffset.FromUnixTimeSeconds(expUnix).UtcDateTime;
                        _log.LogInformation("JWT exp (UTC): {ExpUtc}", expUtc);
                    }
                }
            }
            catch { /* best-effort */ }

            static string PadB64(string s) => s.PadRight(s.Length + (4 - s.Length % 4) % 4, '=').Replace('-', '+').Replace('_', '/');
        }

        public UserHubAgent(ILogger<UserHubAgent> log, StatusService statusService)
        {
            _log = log;
            _statusService = statusService;
        }

        public async Task ConnectAsync(string jwtToken, long accountId, CancellationToken ct)
        {
            LogTokenExp(jwtToken); // Print JWT exp for diagnostics
            await _gate.WaitAsync(ct);
            try
            {
                if (_hub != null && (_hub.State == HubConnectionState.Connected || _hub.State == HubConnectionState.Connecting || _hub.State == HubConnectionState.Reconnecting))
                {
                    _log.LogInformation($"UserHub already in state {_hub.State}");
                    return;
                }

                _hub = new HubConnectionBuilder()
                    .WithUrl("https://rtc.topstepx.com/hubs/user", o =>
                    {
                        o.AccessTokenProvider = () => Task.FromResult<string?>(jwtToken);
                        // DIAGNOSTIC MODE: allow negotiate/fallback while we learn WHY it closes.
                        // o.SkipNegotiation = true;
                        // o.Transports = HttpTransportType.WebSockets;
                    })
                    .WithAutomaticReconnect(new[] { TimeSpan.Zero, TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10) })
                    .ConfigureLogging(b =>
                    {
                        b.AddConsole();
                        b.SetMinimumLevel(LogLevel.Debug);
                        b.AddFilter("Microsoft.AspNetCore.SignalR.Client", LogLevel.Debug);
                        b.AddFilter("Microsoft.AspNetCore.Http.Connections.Client", LogLevel.Debug);
                        b.AddFilter("System.Net.Http", LogLevel.Information);
                    })
                    .Build();

                WireHandlersOnce(); // attach handlers only once

                try
                {
                    await _hub.StartAsync(ct);
                    Console.WriteLine($"UserHub connected. State={_hub.State}");
                    _statusService.Set("user.state", _hub.ConnectionId ?? string.Empty);
                }
                catch (Exception startEx)
                {
                    _log.LogError(startEx, "UserHub StartAsync FAILED. State={State}", _hub.State);
                    throw; // don't proceed to Invoke if start failed
                }

                // Set static _currentHub for diagnostics
                _currentHub = _hub;
                // Use HubSafe for guarded invocation
                await HubSafe.InvokeAsync(_hub, () => _hub.InvokeAsync("JoinAccountRoom", accountId, ct), ct);
                await HubSafe.InvokeAsync(_hub, () => _hub.InvokeAsync("SubscribeUserEvents", accountId, ct), ct);
            }
            finally { _gate.Release(); }
        }

        private void WireHandlersOnce()
        {
            if (_wired) return;

            _hub.Closed += ex =>
            {
                _log.LogInformation($"=== UserHub CLOSED ===\nTYPE: {ex?.GetType().FullName ?? "<null>"}\nMSG : {ex?.Message ?? "<null>"}\nFULL: {ex}");
                if (ex is HttpRequestException hre)
                    _log.LogInformation($"HTTP StatusCode: {hre.StatusCode}");
                if (ex?.InnerException is Exception ie)
                    _log.LogInformation($"INNER: {ie.GetType().FullName} | {ie.Message}");
                return Task.CompletedTask;
            };

            _hub.Reconnecting += ex =>
            {
                _log.LogInformation($"=== UserHub RECONNECTING ===\nTYPE: {ex?.GetType().FullName ?? "<null>"} | MSG: {ex?.Message ?? "<null>"}");
                return Task.CompletedTask;
            };

            _hub.Reconnected += id =>
            {
                _log.LogInformation($"=== UserHub RECONNECTED : {id} ===");
                return Task.CompletedTask;
            };

            _wired = true;
        }

        private static async Task SafeInvokeAsync(Func<Task> call, CancellationToken ct, int maxAttempts = 5)
        {
            for (var a = 1; ; a++)
            {
                await WaitUntilConnectedAsync(_currentHub, ct);
                try { await call(); return; }
                catch (InvalidOperationException ioe) when (ioe.Message.Contains("not active") && a < maxAttempts)
                { await Task.Delay(150 * a, ct); }
            }
        }

        private static HubConnection? _currentHub;
        private static async Task WaitUntilConnectedAsync(HubConnection? hub, CancellationToken ct)
        {
            if (hub is null) return;
            while (hub.State != HubConnectionState.Connected)
            {
                ct.ThrowIfCancellationRequested();
                if (hub.State == HubConnectionState.Disconnected)
                {
                    try { await hub.StartAsync(ct); } catch { /* let reconnect policy do its thing */ }
                }
                await Task.Delay(120, ct);
            }
        }
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