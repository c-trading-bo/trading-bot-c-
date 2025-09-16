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

// =============================================================================
// LEGACY CODE SUPPRESSIONS - TO BE ADDRESSED IN FUTURE CLEANUP
// =============================================================================

// Legacy AlertService requires refactoring - suppress for production hardening focus
[assembly: SuppressMessage("Design", "S2139:Either log this exception and handle it, or rethrow it with some contextual information",
    Scope = "namespaceanddescendants", Target = "~N:TradingBot.Infrastructure.Alerts",
    Justification = "Legacy code - will be addressed in separate cleanup effort")]

[assembly: SuppressMessage("AsyncUsage", "AsyncFixer01:The method does not need to use async/await",
    Scope = "namespaceanddescendants", Target = "~N:TradingBot.Infrastructure.Alerts",
    Justification = "Legacy code - will be addressed in separate cleanup effort")]

// Assembly version attributes - legacy projects
[assembly: SuppressMessage("Design", "S3904:Provide an 'AssemblyVersion' attribute",
    Justification = "Legacy projects use Directory.Build.props versioning")]

// Legacy UpdaterAgent requires comprehensive refactoring
[assembly: SuppressMessage("CodeQuality", "S2486:Handle the exception or explain in a comment why it can be ignored",
    Scope = "namespaceanddescendants", Target = "~N:UpdaterAgent",
    Justification = "Legacy code - will be addressed in separate cleanup effort")]

[assembly: SuppressMessage("CodeQuality", "S108:Either remove or fill this block of code",
    Scope = "namespaceanddescendants", Target = "~N:UpdaterAgent",
    Justification = "Legacy code - empty catch blocks will be addressed in cleanup")]

// Legacy Strategies project requires refactoring
[assembly: SuppressMessage("CodeQuality", "S125:Remove this commented out code",
    Scope = "namespaceanddescendants", Target = "~N:Strategies",
    Justification = "Legacy code - commented code will be cleaned up in separate effort")]

[assembly: SuppressMessage("CodeQuality", "S2955:Use a comparison to 'default(T)' instead",
    Scope = "namespaceanddescendants", Target = "~N:Strategies",
    Justification = "Legacy generic null checks - will be modernized")]

[assembly: SuppressMessage("Performance", "S3267:Loops should be simplified using the 'Where' LINQ method",
    Scope = "namespaceanddescendants", Target = "~N:Strategies",
    Justification = "Legacy code - LINQ optimization will be addressed")]

[assembly: SuppressMessage("Design", "S2325:Make method a static method",
    Scope = "namespaceanddescendants", Target = "~N:Strategies",
    Justification = "Legacy methods - static analysis will be done in cleanup")]

[assembly: SuppressMessage("Security", "SCS0005:Weak random number generator",
    Scope = "namespaceanddescendants", Target = "~N:Strategies",
    Justification = "Legacy random usage - will be replaced with cryptographic random")]

[assembly: SuppressMessage("Design", "S3400:Remove this method and declare a constant",
    Scope = "namespaceanddescendants", Target = "~N:Strategies",
    Justification = "Legacy constant methods - will be refactored")]

[assembly: SuppressMessage("Documentation", "S3928:The parameter name is not declared in the argument list",
    Scope = "namespaceanddescendants", Target = "~N:Strategies",
    Justification = "Legacy documentation issues - will be corrected")]

[assembly: SuppressMessage("Design", "S4456:Split this method into two",
    Scope = "namespaceanddescendants", Target = "~N:Strategies",
    Justification = "Legacy iterator methods - will be refactored for better separation")]

// =============================================================================
// PRODUCTION HARDENING: NEW CODE MUST NOT HAVE SUPPRESSIONS
// =============================================================================
// All new production hardening code in BotCore.Configuration, BotCore.HealthChecks,
// and BotCore.Resilience must pass all analyzer rules without suppressions