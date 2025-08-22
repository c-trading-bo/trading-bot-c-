
#nullable enable
using System;
using System.Collections.Generic;
using BotCore.Models;

namespace BotCore
{
    public sealed class StrategyContext
    {
        public int AccountId { get; }
        public IReadOnlyDictionary<string,string> ContractIds { get; } // e.g. ES -> CON.F.US.EP.U25
        public Func<string, IReadOnlyList<Bar>> GetBars { get; }      // pull bars for any symbol
        public Action<string,string,string>? Log { get; set; }         // (level,strategy,message)

        public StrategyContext(int accountId,
                               IReadOnlyDictionary<string,string> contractIds,
                               Func<string, IReadOnlyList<Bar>> getBars)
        {
            AccountId = accountId;
            ContractIds = contractIds;
            GetBars = getBars;
        }
    }
}
