using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using BotCore;

namespace OrchestratorAgent.Ops
{
    public sealed class EodReconciler
    {
        private readonly ApiClient _api;
        private readonly long _accountId;
        private readonly string _tzId;
        private readonly TimeSpan _settleLocal;
        private readonly string _journalPath;

        public EodReconciler(ApiClient api, long accountId, string tzId, string settleTimeLocalHHmm, string journalDir = "state")
        {
            _api = api; _accountId = accountId; _tzId = tzId;
            var parts = (settleTimeLocalHHmm ?? "15:00").Split(':');
            _settleLocal = new TimeSpan(int.Parse(parts[0]), int.Parse(parts[1]), 0);
            Directory.CreateDirectory(journalDir);
            _journalPath = Path.Combine(journalDir, "eod_journal.jsonl");
        }

        public async Task RunLoopAsync(Func<Task> resetCounters, CancellationToken ct)
        {
            var tz = ResolveTz(_tzId);
            DateTime lastRunDateLocal = DateTime.MinValue;

            while (!ct.IsCancellationRequested)
            {
                try
                {
                    var nowLocal = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, tz);
                    if (nowLocal.TimeOfDay >= _settleLocal && nowLocal.Date != lastRunDateLocal.Date)
                    {
                        decimal? net = null, gross = null, fees = null;
                        try
                        {
                            var pnl = await _api.GetAsync<JsonElement>($"/accounts/{_accountId}/pnl?scope=today", ct);
                            if (pnl.ValueKind == JsonValueKind.Object)
                            {
                                if (pnl.TryGetProperty("net", out var n) && n.TryGetDecimal(out var nd)) net = nd;
                                if (pnl.TryGetProperty("gross", out var g) && g.TryGetDecimal(out var gd)) gross = gd;
                                if (pnl.TryGetProperty("fees", out var f) && f.TryGetDecimal(out var fd)) fees = fd;
                            }
                        }
                        catch { }
                        var rec = new { utc = DateTime.UtcNow, local = nowLocal, Net = net, Gross = gross, Fees = fees };
                        await File.AppendAllTextAsync(_journalPath, JsonSerializer.Serialize(rec) + Environment.NewLine, ct);

                        if (resetCounters != null) await resetCounters();
                        lastRunDateLocal = nowLocal.Date;
                    }
                }
                catch (OperationCanceledException) { }
                catch { }

                try { await Task.Delay(TimeSpan.FromSeconds(20), ct); } catch { }
            }
        }

        private static TimeZoneInfo ResolveTz(string tzId)
        {
            try { return TimeZoneInfo.FindSystemTimeZoneById(tzId); } catch { }
            try { return TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time"); } catch { }
            return TimeZoneInfo.Utc;
        }
    }
}
