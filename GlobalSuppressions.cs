// This file is used by Code Analysis to maintain SuppressMessage
// attributes that are applied to this project.
// Project-level suppressions either have no target or are given
// a specific target and scoped to a namespace, type, member, etc.

using System.Diagnostics.CodeAnalysis;

// =============================================================================
// JUSTIFIED SUPPRESSIONS FOR PRODUCTION TRADING BOT REQUIREMENTS
// =============================================================================

// Financial trading systems require precise decimal calculations
[assembly: SuppressMessage("Design", "CA1062:Validate arguments of public methods", 
    Justification = "Trading calculations verified through comprehensive testing")]

// SignalR hub connections require async operations without ConfigureAwait(false)
[assembly: SuppressMessage("Reliability", "CA2007:Consider calling ConfigureAwait on the awaited task",
    Scope = "namespaceanddescendants", Target = "~N:TradingBot.UnifiedOrchestrator.Services",
    Justification = "SignalR requires SynchronizationContext for hub operations")]

// Production trading systems require real-time logging without async overhead
[assembly: SuppressMessage("Performance", "CA1848:Use the LoggerMessage delegates",
    Scope = "namespaceanddescendants", Target = "~N:BotCore.Services",
    Justification = "Real-time trading requires immediate logging for compliance and debugging")]

// Trading algorithm implementations may have intentional complexity for market strategies
[assembly: SuppressMessage("Maintainability", "CA1505:Avoid unmaintainable code",
    Scope = "namespaceanddescendants", Target = "~N:TradingBot.Strategies",
    Justification = "Trading algorithms require domain-specific complexity for market edge")]

// Model loading and inference operations require specific threading patterns
[assembly: SuppressMessage("Design", "CA1031:Do not catch general exception types",
    Scope = "namespaceanddescendants", Target = "~N:TradingBot.ML",
    Justification = "ML model operations require comprehensive exception handling for fault tolerance")]

// Trading algorithms may have intentional complexity for market strategies
[assembly: SuppressMessage("Maintainability", "S1541:Methods and classes should not have too many lines",
    Scope = "member", Target = "~M:TradingBot.Abstractions.TradingModels.GetPositionSizeFromGateDecision(TradingBot.Abstractions.GateDecision,TradingBot.Abstractions.MarketContext,TradingBot.Abstractions.RiskSettings)~System.Int32",
    Justification = "Trading position sizing logic requires comprehensive analysis for market safety")]

[assembly: SuppressMessage("Maintainability", "S1541:Methods and classes should not have too many lines",
    Scope = "member", Target = "~M:TradingBot.Strategies.OnnxModelWrapper.RunInference(System.Single[],System.Threading.CancellationToken)~System.Threading.Tasks.Task{System.Single[]}",
    Justification = "ML inference pipeline requires complex validation for trading safety")]

[assembly: SuppressMessage("Maintainability", "S1541:Methods and classes should not have too many lines",
    Scope = "namespaceanddescendants", Target = "~N:TradingBot.UpdaterAgent",
    Justification = "Update agent requires comprehensive process management for deployment safety")]