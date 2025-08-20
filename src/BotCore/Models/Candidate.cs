using System;
using BotCore.Models;

namespace BotCore.Models
{
    public sealed class Candidate
    {
    public string strategy_id { get; set; } = "";
    public string symbol { get; set; } = "";
    public Side side { get; set; }
    public decimal entry { get; set; }
    public decimal stop { get; set; }
    public decimal t1 { get; set; }
    public decimal expR { get; set; }
    public decimal qty { get; set; }
    public bool atr_ok { get; set; }
    public decimal? vol_z { get; set; }
    public long accountId { get; set; }
    public string contractId { get; set; } = "";
    }
}
