#nullable enable
using System.Text.Json;

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
            catch { return new Snapshot(); }
        }

        public void Save(Snapshot snap)
        {
            try
            {
                var json = JsonSerializer.Serialize(snap, JsonOptions);
                File.WriteAllText(_path, json);
            }
            catch { }
        }
    }
}
