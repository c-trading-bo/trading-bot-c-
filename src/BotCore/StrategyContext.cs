
#nullable enable
using System;
using System.Collections.Generic;
using BotCore.Models;

namespace BotCore
{
    public sealed class StrategyContext(int accountId,
                           IReadOnlyDictionary<string, string> contractIds,
                           Func<string, IReadOnlyList<Bar>> getBars)
    {
        public int AccountId { get; } = accountId;
        public IReadOnlyDictionary<string, string> ContractIds { get; } = contractIds;
        public Func<string, IReadOnlyList<Bar>> GetBars { get; } = getBars;
        public Action<string, string, string>? Log { get; set; }         // (level,strategy,message)
    }
}
