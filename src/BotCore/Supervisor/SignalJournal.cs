#nullable enable
using System;
using System.Collections.Concurrent;
using System.IO;
using System.Text;
using BotCore.Models;
using System.Globalization;

namespace BotCore.Supervisor
{
    /// <summary>Minimal CSV journal for emitted/gated/routed signals for postmortem.</summary>
    public sealed class SignalJournal
    {
        private readonly string _path;
        private static readonly string Header = "utc,strategy,symbol,side,entry,stop,target,qty,score,stage,cid,note";
        private readonly object _sync = new();

        public SignalJournal(string? filePath = null)
        {
            _path = filePath ?? Path.Combine(AppContext.BaseDirectory, "signals.csv");
            EnsureHeader();
        }

        private void EnsureHeader()
        {
            lock (_sync)
            {
                if (!File.Exists(_path))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
                    File.WriteAllText(_path, Header + Environment.NewLine);
                }
            }
        }

        public void Append(BotCore.Models.Signal s, string stage, string? cid = null)
        {
            var line = string.Join(',', [
                DateTime.UtcNow.ToString("o", CultureInfo.InvariantCulture),
                s.StrategyId,
                s.Symbol,
                s.Side,
                s.Entry.ToString(CultureInfo.InvariantCulture),
                s.Stop.ToString(CultureInfo.InvariantCulture),
                s.Target.ToString(CultureInfo.InvariantCulture),
                s.Size.ToString(CultureInfo.InvariantCulture),
                s.Score.ToString(CultureInfo.InvariantCulture),
                stage,
                cid ?? string.Empty,
                s.Tag.Replace(',', ';')
            ]);
            lock (_sync)
            {
                File.AppendAllText(_path, line + Environment.NewLine, Encoding.UTF8);
            }
        }
    }
}
