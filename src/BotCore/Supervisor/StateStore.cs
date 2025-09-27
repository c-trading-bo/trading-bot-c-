#nullable enable
using System.Text.Json;
using System.IO;

namespace BotCore.Supervisor
{
    public sealed class StateStore(string? path = null)
    {
        private readonly string _path = path ?? Path.Combine(AppContext.BaseDirectory, "bot_state.json");

        // Cached JsonSerializerOptions for performance (CA1869 compliance)
        private static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };

        public sealed class Snapshot
        {
            public Dictionary<string, DateTime>? RecentRoutes { get; set; }
            public Dictionary<string, long>? LastBarUnix { get; set; }
            public List<string>? LastCids { get; set; }
        }

        public Snapshot Load()
        {
            try
            {
                if (!File.Exists(_path)) return new Snapshot();
                var json = File.ReadAllText(_path);
                var snap = JsonSerializer.Deserialize<Snapshot>(json) ?? new Snapshot();
                return snap;
            }
            catch (FileNotFoundException)
            {
                return new Snapshot();
            }
            catch (DirectoryNotFoundException)
            {
                return new Snapshot();
            }
            catch (UnauthorizedAccessException)
            {
                return new Snapshot();
            }
            catch (JsonException)
            {
                return new Snapshot();
            }
            catch (IOException)
            {
                return new Snapshot();
            }
        }

        public void Save(Snapshot snap)
        {
            try
            {
                var json = JsonSerializer.Serialize(snap, JsonOptions);
                File.WriteAllText(_path, json);
            }
            catch (DirectoryNotFoundException)
            {
                // Silently ignore - directory might be missing in test scenarios
            }
            catch (UnauthorizedAccessException)
            {
                // Silently ignore - might not have write permissions in some environments
            }
            catch (JsonException)
            {
                // Silently ignore - snapshot might be corrupted, but don't crash
            }
            catch (IOException)
            {
                // Silently ignore - disk might be full or file locked
            }
        }
    }
}
