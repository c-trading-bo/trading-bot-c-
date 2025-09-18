using System;
using System.IO;
using System.Text;
using System.Threading.Tasks;

namespace OrchestratorAgent.Ops
{
    /// <summary>Single-writer lease: only one process can route orders (exclusive lock on a file).</summary>
    public sealed class LiveLease : IAsyncDisposable
    {
        private readonly string _path = null!;
        private FileStream? _stream;
        public bool HasLease => _stream is not null;
        public string HolderId { get; } = $"{Environment.MachineName}:{Environment.ProcessId}";

        public LiveLease(string path = "state/live.lock")
        {
            _path = Path.GetFullPath(path);
            Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
        }

        public async Task<bool> TryAcquireAsync()
        {
            if (HasLease) return true;
            try
            {
                _stream = new FileStream(_path, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
                var bytes = Encoding.UTF8.GetBytes(HolderId);
                _stream.SetLength(0);
                await _stream.WriteAsync(bytes).ConfigureAwait(false);
                await _stream.FlushAsync().ConfigureAwait(false);
                return true;
            }
            catch (IOException) { return false; } // held by another process
        }

        public async Task ReleaseAsync()
        {
            if (_stream is null) return;
            try { await _stream.FlushAsync().ConfigureAwait(false); } catch { /* ignore */ }
            _stream.Dispose(); _stream = null!;
            try { File.Delete(_path); } catch { /* ignore */ }
        }

        public async ValueTask DisposeAsync() => await ReleaseAsync().ConfigureAwait(false);
    }
}
