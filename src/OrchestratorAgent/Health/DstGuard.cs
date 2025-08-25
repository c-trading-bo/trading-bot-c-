using System;

namespace OrchestratorAgent.Health
{
    public sealed class DstGuard
    {
        private readonly TimeZoneInfo _tz;
        private readonly int _warnWindowDays;
        private TimeSpan _lastOffset;

        public DstGuard(string tzId = "America/Chicago", int warnWindowDays = 7)
        {
            _tz = ResolveTz(tzId);
            _warnWindowDays = warnWindowDays;
            var now = DateTime.UtcNow;
            _lastOffset = TimeZoneInfo.ConvertTime(now, _tz) - now;
        }

        public (bool ok, string? warn) Check()
        {
            var now = DateTime.UtcNow;
            var offset = TimeZoneInfo.ConvertTime(now, _tz) - now;

            var changedRecently = false;
            for (int d = 0; d <= _warnWindowDays; d++)
            {
                var dt = now.AddDays(-d);
                var o = TimeZoneInfo.ConvertTime(dt, _tz) - dt;
                if (o != offset) { changedRecently = true; break; }
            }

            var shifted = offset != _lastOffset;
            _lastOffset = offset;

            if (changedRecently || shifted)
                return (true, $"DST boundary window active. Exchange offset now {(int)offset.TotalHours}h vs UTC. Verify session template.");

            return (true, null);
        }

        public bool IsOpen(DateTime utc, (int openHour, int openMin, int closeHour, int closeMin) sess)
        {
            var local = TimeZoneInfo.ConvertTimeFromUtc(utc, _tz);
            var open  = new DateTime(local.Year, local.Month, local.Day, sess.openHour,  sess.openMin, 0, local.Kind);
            var close = new DateTime(local.Year, local.Month, local.Day, sess.closeHour, sess.closeMin, 0, local.Kind);
            if (close <= open) close = close.AddDays(1);
            return local >= open && local < close;
        }

        private static TimeZoneInfo ResolveTz(string tzId)
        {
            try { return TimeZoneInfo.FindSystemTimeZoneById(tzId); } catch { }
            // Windows fallback
            try { return TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time"); } catch { }
            return TimeZoneInfo.Utc;
        }
    }
}
