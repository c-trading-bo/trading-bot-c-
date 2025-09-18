#nullable enable
using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;

namespace OrchestratorAgent.Infra
{
    public sealed class Notifier
    {
        private readonly HttpClient _http = new();
        private readonly string? _url = Environment.GetEnvironmentVariable("BOT_ALERT_WEBHOOK");
        public Task Info(string m) => Send("INFO", m);
        public Task Warn(string m) => Send("WARN", m);
        public Task Error(string m) => Send("ERROR", m);
        private async Task Send(string lvl, string m)
        {
            if (string.IsNullOrWhiteSpace(_url)) return;
            var payload = new { content = $"`{lvl}` {DateTime.UtcNow:HH:mm:ss} {m}" };
            try { await _http.PostAsJsonAsync(_url, payload).ConfigureAwait(false); } catch { }
        }
    }
}
