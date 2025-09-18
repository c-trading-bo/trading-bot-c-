using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Realtime
{
    public sealed class ReplayFeeder
    {
        public static async IAsyncEnumerable<(string type, JsonElement data)> ReadAsync(string path, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct)
        {
            using var r = new StreamReader(File.OpenRead(path));
            string? line;
            while ((line = await r.ReadLineAsync(ct)) != null)
            {
                ct.ThrowIfCancellationRequested().ConfigureAwait(false);
                using var doc = JsonDocument.Parse(line);
                yield return (doc.RootElement.GetProperty("type").GetString()!,
                              doc.RootElement.GetProperty("data"));
            }
        }
    }
}
