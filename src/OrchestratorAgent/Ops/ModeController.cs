using System;
using System.Text.Json;

namespace OrchestratorAgent.Ops
{
    public enum TradeMode { Shadow = 0, Live = 1 }

    public sealed class ModeController
    {
        private readonly string _statePath = System.IO.Path.Combine(AppContext.BaseDirectory, "state", "mode.json");
        public TradeMode Mode { get; private set; } = TradeMode.Shadow;
        public event Action<TradeMode>? OnChange;

        public ModeController(bool stickyLive)
        {
            if (!stickyLive) return;
            try
            {
                if (System.IO.File.Exists(_statePath))
                {
                    var m = JsonSerializer.Deserialize<TradeMode>(System.IO.File.ReadAllText(_statePath));
                    if (Enum.IsDefined(typeof(TradeMode), m)) Mode = m;
                }
            }
            catch { /* ignore */ }
        }

        public void Set(TradeMode m)
        {
            if (Mode == m) return;
            Mode = m;
            try
            {
                System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(_statePath)!);
                System.IO.File.WriteAllText(_statePath, JsonSerializer.Serialize(m));
            }
            catch { /* ignore */ }
            OnChange?.Invoke(m);
        }

        public bool IsLive => Mode == TradeMode.Live;
    }
}
