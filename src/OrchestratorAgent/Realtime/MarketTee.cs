using System.IO;
using System.Text.Json;

namespace OrchestratorAgent.Realtime
{
    internal sealed class MarketTee : IAsyncDisposable
    {
        private readonly StreamWriter? _writer;
        private static readonly JsonSerializerOptions _json = new(JsonSerializerDefaults.Web);

        public MarketTee(string? path)
        {
            if (string.IsNullOrWhiteSpace(path)) return;
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            _writer = new StreamWriter(File.Open(path, FileMode.Append, FileAccess.Write, FileShare.Read));
        }

        public void OnQuote(object quote) => Write("quote", quote);
        public void OnTrade(object trade) => Write("trade", trade);

        private void Write(string type, object obj)
        {
            if (_writer is null) return;
            var line = JsonSerializer.Serialize(new { type, tsUtc = DateTimeOffset.UtcNow, data = obj }, _json);
            _writer.WriteLine(line);
        }

        public async ValueTask DisposeAsync()
        {
            if (_writer != null) { await _writer.FlushAsync().ConfigureAwait(false); _writer.Dispose(); }
        }
    }
}
