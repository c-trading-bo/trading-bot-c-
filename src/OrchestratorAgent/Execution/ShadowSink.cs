using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Execution
{
    internal sealed class ShadowSink(string path = "logs/shadow-orders.ndjson") : IExecutionSink
    {
        private readonly string _path = path;
        private static readonly JsonSerializerOptions _json = new(JsonSerializerDefaults.Web);

        public Task<object> HandleAsync(NewOrder o, CancellationToken ct)
        {
            var payload = new { mode = "shadow", wouldSend = o, tsUtc = DateTimeOffset.UtcNow };
            Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
            File.AppendAllText(_path, JsonSerializer.Serialize(payload, _json) + System.Environment.NewLine);
            return Task.FromResult<object>(new { status = "shadowed" });
        }
    }
}
