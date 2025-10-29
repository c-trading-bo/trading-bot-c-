using Microsoft.Extensions.Logging;
using BotCore.Market;
using Zones;
using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Features;

/// <summary>
/// Session-specific parameters resolver
/// Adjusts trading parameters based on session (Asian/London/NY/Overnight)
/// Ensures historical backtesting respects session-specific risk and sizing rules
/// </summary>
public sealed class SessionParametersResolver : IFeatureResolver
{
    private readonly ILogger<SessionParametersResolver> _logger;
    private readonly IFeatureBus _featureBus;
    private readonly ConcurrentDictionary<string, double> _latestFeatures = new();

    private static readonly string[] FeatureKeys = new[]
    {
        "session.type",
        "session.volatility_mult",
        "session.size_mult",
        "session.stop_mult",
        "session.target_mult"
    };

    public SessionParametersResolver(
        ILogger<SessionParametersResolver> logger,
        IFeatureBus featureBus)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _featureBus = featureBus ?? throw new ArgumentNullException(nameof(featureBus));
    }

    public Task OnBarAsync(string symbol, object barData, CancellationToken cancellationToken = default)
    {
        try
        {
            if (barData is not Bar bar)
            {
                _logger.LogWarning("[SESSION-PARAMS] Invalid bar data type for {Symbol}", symbol);
                return Task.CompletedTask;
            }

            var session = DetectSession(bar.Start);
            var parameters = GetSessionParameters(session);

            var now = DateTime.UtcNow;

            _featureBus.Publish(symbol, now, "session.type", (double)session);
            _featureBus.Publish(symbol, now, "session.volatility_mult", parameters.VolatilityMultiplier);
            _featureBus.Publish(symbol, now, "session.size_mult", parameters.SizeMultiplier);
            _featureBus.Publish(symbol, now, "session.stop_mult", parameters.StopMultiplier);
            _featureBus.Publish(symbol, now, "session.target_mult", parameters.TargetMultiplier);

            var key = symbol;
            _latestFeatures[$"{key}::session.type"] = (double)session;
            _latestFeatures[$"{key}::session.volatility_mult"] = parameters.VolatilityMultiplier;
            _latestFeatures[$"{key}::session.size_mult"] = parameters.SizeMultiplier;
            _latestFeatures[$"{key}::session.stop_mult"] = parameters.StopMultiplier;
            _latestFeatures[$"{key}::session.target_mult"] = parameters.TargetMultiplier;

            _logger.LogTrace("[SESSION-PARAMS] {Symbol} @ {Time}: Session={Session}, SizeMult={Size:F2}",
                symbol, bar.Start, session, parameters.SizeMultiplier);

            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SESSION-PARAMS] Error processing bar for {Symbol}", symbol);
            throw;
        }
    }

    public Task<double?> TryGetAsync(string symbol, string featureKey, CancellationToken cancellationToken = default)
    {
        var key = $"{symbol}::{featureKey}";
        if (_latestFeatures.TryGetValue(key, out var value))
        {
            return Task.FromResult<double?>(value);
        }
        return Task.FromResult<double?>(null);
    }

    public string[] GetAvailableFeatureKeys() => FeatureKeys;

    private TradingSession DetectSession(DateTime timestamp)
    {
        var utcHour = timestamp.Hour;

        if (utcHour >= 23 || utcHour < 7)
            return TradingSession.Asian;

        if (utcHour >= 8 && utcHour < 16)
            return TradingSession.London;

        if (utcHour >= 14 && utcHour < 21)
            return TradingSession.NewYork;

        return TradingSession.Overnight;
    }

    private SessionParameters GetSessionParameters(TradingSession session)
    {
        return session switch
        {
            TradingSession.Asian => new SessionParameters
            {
                VolatilityMultiplier = 0.7,
                SizeMultiplier = 0.5,
                StopMultiplier = 1.2,
                TargetMultiplier = 0.8
            },
            TradingSession.London => new SessionParameters
            {
                VolatilityMultiplier = 1.3,
                SizeMultiplier = 1.0,
                StopMultiplier = 1.0,
                TargetMultiplier = 1.2
            },
            TradingSession.NewYork => new SessionParameters
            {
                VolatilityMultiplier = 1.5,
                SizeMultiplier = 1.2,
                StopMultiplier = 0.9,
                TargetMultiplier = 1.5
            },
            TradingSession.Overnight => new SessionParameters
            {
                VolatilityMultiplier = 0.5,
                SizeMultiplier = 0.3,
                StopMultiplier = 1.5,
                TargetMultiplier = 0.6
            },
            _ => new SessionParameters
            {
                VolatilityMultiplier = 1.0,
                SizeMultiplier = 1.0,
                StopMultiplier = 1.0,
                TargetMultiplier = 1.0
            }
        };
    }

    private enum TradingSession
    {
        Asian = 0,
        London = 1,
        NewYork = 2,
        Overnight = 3
    }

    private class SessionParameters
    {
        public double VolatilityMultiplier { get; set; }
        public double SizeMultiplier { get; set; }
        public double StopMultiplier { get; set; }
        public double TargetMultiplier { get; set; }
    }
}
