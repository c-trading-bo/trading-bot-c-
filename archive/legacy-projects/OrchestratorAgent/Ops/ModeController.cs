using System;
using System.Text.Json;

namespace OrchestratorAgent.Ops
{
    internal enum TradeMode { Dry = 0, Shadow = 1, Live = 2 }

    internal sealed class ModeController
    {
        private readonly string _statePath = System.IO.Path.Combine(AppContext.BaseDirectory, "state", "mode.json");
        public TradeMode Mode { get; private set; } = TradeMode.Shadow;
        public event Action<TradeMode>? OnChange;
        public bool Autopilot { get; }

        public ModeController(bool stickyLive, bool autopilot = false)
        {
            Autopilot = autopilot;
            if (stickyLive)
            {
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
            SyncLegacyEnv();
        }

        private void SyncLegacyEnv()
        {
            // Mirror for any legacy code looking at LIVE_ORDERS
            Environment.SetEnvironmentVariable("LIVE_ORDERS", IsLive ? "1" : "0");
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
            SyncLegacyEnv();
            OnChange?.Invoke(m);
        }

        public bool IsLive => Mode == TradeMode.Live;
        public bool IsShadow => Mode == TradeMode.Shadow;
        public bool IsDry => Mode == TradeMode.Dry;
    }
}
