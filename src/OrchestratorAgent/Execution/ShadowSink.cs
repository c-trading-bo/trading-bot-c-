using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Execution
{
    public sealed class ShadowSink : IExecutionSink
    {
        private readonly string _path;
        private static readonly JsonSerializerOptions _json = new(JsonSerializerDefaults.Web);

        public ShadowSink(string path = "logs/shadow-orders.ndjson") => _path = path;

        public Task<object> HandleAsync(NewOrder o, CancellationToken ct)
        {
            var payload = new { mode = "shadow", wouldSend = o, tsUtc = DateTimeOffset.UtcNow };
            Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
            File.AppendAllText(_path, JsonSerializer.Serialize(payload, _json) + System.Environment.NewLine);
            return Task.FromResult<object>(new { status = "shadowed" });
        }
    }
}
