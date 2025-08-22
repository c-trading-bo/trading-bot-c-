
#nullable enable
using System.Collections.Generic;
using BotCore.Models;
using BotCore;

namespace BotCore
{
    /// <summary>Implement this on each strategy you want to run.</summary>
    public interface IStrategy
    {
        /// <summary>A short, unique name (for logs, risk, routing).</summary>
        string Name { get; }

        /// <summary>Called on each completed bar for a symbol. Return zero or more signals.</summary>
        IEnumerable<StrategySignal> OnBar(string symbol, IReadOnlyList<Bar> bars, StrategyContext ctx);
    }
}
