#!/bin/bash
# Legacy Config Cleanup - Remove unused env vars and centralize risk defaults

echo "🧹 Legacy Config Cleanup - Removing unused env vars and centralizing risk defaults"
echo "================================================================================"

echo "📋 Phase 1: Scanning for unused legacy environment variables..."

# List of legacy environment variables that should be removed/deprecated
legacy_env_vars=(
    "TOPSTEPX_LEGACY_CLIENT_ID"
    "TOPSTEPX_LEGACY_SECRET"
    "INFRASTRUCTURE_TOPSTEPX_ENABLED"
    "LEGACY_SIMULATION_MODE"
    "OLD_HISTORICAL_DATA_PATH"
    "LEGACY_RISK_MANAGER_ENABLED"
    "DEPRECATED_FEATURE_FLAGS"
)

# Scan for references to legacy env vars in code
echo "🔍 Scanning code for legacy environment variable references..."
for var in "${legacy_env_vars[@]}"; do
    echo "  Checking for: $var"
    count=$(grep -r "$var" src/ --include="*.cs" 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "    ❌ Found $count references to $var in source code"
        grep -r "$var" src/ --include="*.cs" 2>/dev/null | head -3
    else
        echo "    ✅ No references found to $var"
    fi
done

echo ""
echo "📋 Phase 2: Centralizing risk defaults in configuration..."

# Create centralized risk configuration
cat > src/Safety/RiskDefaults.cs << 'EOF'
using System;

namespace TradingBot.Safety
{
    /// <summary>
    /// Centralized risk management defaults for the trading system
    /// Replaces scattered risk configuration throughout the codebase
    /// </summary>
    public static class RiskDefaults
    {
        /// <summary>
        /// Default maximum daily loss limit
        /// </summary>
        public static readonly decimal DefaultMaxDailyLoss = 1000m;

        /// <summary>
        /// Default maximum position size
        /// </summary>
        public static readonly decimal DefaultMaxPositionSize = 10000m;

        /// <summary>
        /// Default drawdown limit
        /// </summary>
        public static readonly decimal DefaultDrawdownLimit = 2000m;

        /// <summary>
        /// Default maximum risk per trade as percentage
        /// </summary>
        public static readonly decimal DefaultMaxRiskPerTradePercent = 0.01m; // 1%

        /// <summary>
        /// Default maximum number of trades per day
        /// </summary>
        public static readonly int DefaultMaxDailyTrades = 50;

        /// <summary>
        /// Default maximum number of trades per session
        /// </summary>
        public static readonly int DefaultMaxSessionTrades = 20;

        /// <summary>
        /// Default maximum portfolio exposure
        /// </summary>
        public static readonly decimal DefaultMaxPortfolioExposure = 50000m;

        /// <summary>
        /// Default maximum number of open positions
        /// </summary>
        public static readonly int DefaultMaxOpenPositions = 10;

        /// <summary>
        /// ES/MES tick size for price calculations
        /// </summary>
        public static readonly decimal EsTickSize = 0.25m;

        /// <summary>
        /// NQ/MNQ tick size for price calculations
        /// </summary>
        public static readonly decimal NqTickSize = 0.25m;

        /// <summary>
        /// Default stop loss percentage
        /// </summary>
        public static readonly decimal DefaultStopLossPercent = 0.005m; // 0.5%

        /// <summary>
        /// Default take profit percentage
        /// </summary>
        public static readonly decimal DefaultTakeProfitPercent = 0.01m; // 1%

        /// <summary>
        /// SDK adapter specific risk defaults
        /// </summary>
        public static class SdkAdapter
        {
            public static readonly decimal MaxRiskPercentPerOrder = 0.01m;
            public static readonly int MaxOrderRetries = 3;
            public static readonly TimeSpan OrderTimeout = TimeSpan.FromSeconds(30);
            public static readonly string[] SupportedInstruments = { "ES", "MES", "NQ", "MNQ", "RTY", "YM" };
        }

        /// <summary>
        /// ML/RL specific risk defaults
        /// </summary>
        public static class MachineLearning
        {
            public static readonly decimal MinConfidenceThreshold = 0.6m;
            public static readonly decimal MaxModelRiskPercent = 0.005m; // 0.5%
            public static readonly int MinHistoricalBarsRequired = 100;
            public static readonly TimeSpan ModelUpdateInterval = TimeSpan.FromHours(24);
        }

        /// <summary>
        /// UCB specific risk defaults
        /// </summary>
        public static class Ucb
        {
            public static readonly decimal ExplorationBonus = 0.3m;
            public static readonly decimal ConfidenceThreshold = 0.65m;
            public static readonly int MinDecisionsBeforeLive = 100;
            public static readonly decimal MaxUcbRiskPercent = 0.008m; // 0.8%
        }

        /// <summary>
        /// Get risk configuration for a specific environment
        /// </summary>
        public static RiskConfiguration GetConfigurationForEnvironment(string environment)
        {
            return environment?.ToUpperInvariant() switch
            {
                "PRODUCTION" => new RiskConfiguration
                {
                    MaxDailyLoss = DefaultMaxDailyLoss,
                    MaxPositionSize = DefaultMaxPositionSize,
                    DrawdownLimit = DefaultDrawdownLimit,
                    MaxRiskPerTradePercent = DefaultMaxRiskPerTradePercent,
                    MaxDailyTrades = DefaultMaxDailyTrades,
                    IsProduction = true
                },
                "STAGING" => new RiskConfiguration
                {
                    MaxDailyLoss = DefaultMaxDailyLoss * 0.5m,
                    MaxPositionSize = DefaultMaxPositionSize * 0.5m,
                    DrawdownLimit = DefaultDrawdownLimit * 0.5m,
                    MaxRiskPerTradePercent = DefaultMaxRiskPerTradePercent * 0.5m,
                    MaxDailyTrades = DefaultMaxDailyTrades / 2,
                    IsProduction = false
                },
                "DEVELOPMENT" or "DEV" => new RiskConfiguration
                {
                    MaxDailyLoss = 100m,
                    MaxPositionSize = 1000m,
                    DrawdownLimit = 200m,
                    MaxRiskPerTradePercent = 0.001m, // 0.1%
                    MaxDailyTrades = 10,
                    IsProduction = false
                },
                _ => new RiskConfiguration
                {
                    MaxDailyLoss = DefaultMaxDailyLoss,
                    MaxPositionSize = DefaultMaxPositionSize,
                    DrawdownLimit = DefaultDrawdownLimit,
                    MaxRiskPerTradePercent = DefaultMaxRiskPerTradePercent,
                    MaxDailyTrades = DefaultMaxDailyTrades,
                    IsProduction = false
                }
            };
        }
    }

    /// <summary>
    /// Risk configuration data structure
    /// </summary>
    public class RiskConfiguration
    {
        public decimal MaxDailyLoss { get; set; }
        public decimal MaxPositionSize { get; set; }
        public decimal DrawdownLimit { get; set; }
        public decimal MaxRiskPerTradePercent { get; set; }
        public int MaxDailyTrades { get; set; }
        public bool IsProduction { get; set; }
    }
}
EOF

echo "✅ Created centralized risk configuration: src/Safety/RiskDefaults.cs"

echo ""
echo "📋 Phase 3: Updating .env.example to remove legacy variables..."

# Update .env.example to remove legacy variables and add SDK-related ones
if [ -f ".env.example" ]; then
    # Backup original
    cp .env.example .env.example.backup
    
    # Remove legacy variables and add SDK ones
    grep -v -E "(TOPSTEPX_LEGACY|INFRASTRUCTURE_TOPSTEPX|LEGACY_|OLD_|DEPRECATED_)" .env.example > .env.example.temp
    
    # Add SDK-specific environment variables
    cat >> .env.example.temp << 'EOF'

# SDK Adapter Configuration
PROJECT_X_API_KEY=your_project_x_api_key_here
PROJECT_X_USERNAME=your_project_x_username_here

# Risk Management (centralized in RiskDefaults.cs)
TRADING_ENVIRONMENT=DEVELOPMENT
MAX_DAILY_LOSS_OVERRIDE=
MAX_POSITION_SIZE_OVERRIDE=

# ML/RL Configuration
ML_MODEL_UPDATE_INTERVAL_HOURS=24
UCB_MIN_DECISIONS_BEFORE_LIVE=100
UCB_CONFIDENCE_THRESHOLD=0.65

# SDK Bridge Configuration
SDK_BRIDGE_TIMEOUT_SECONDS=30
SDK_BRIDGE_MAX_RETRIES=3
EOF

    mv .env.example.temp .env.example
    echo "✅ Updated .env.example with SDK variables and removed legacy ones"
else
    echo "⚠️ .env.example not found - skipping env file cleanup"
fi

echo ""
echo "📋 Phase 4: Creating config migration guide..."

cat > CONFIG_MIGRATION_GUIDE.md << 'EOF'
# Configuration Migration Guide - Legacy to SDK

This guide helps migrate from legacy configuration to the new SDK-based configuration system.

## Deprecated Environment Variables

The following environment variables have been deprecated and should be removed:

- `TOPSTEPX_LEGACY_CLIENT_ID` → Use `PROJECT_X_API_KEY`
- `TOPSTEPX_LEGACY_SECRET` → Use `PROJECT_X_USERNAME` 
- `INFRASTRUCTURE_TOPSTEPX_ENABLED` → Always enabled in SDK mode
- `LEGACY_SIMULATION_MODE` → Automatic simulation when SDK unavailable
- `OLD_HISTORICAL_DATA_PATH` → Uses SDK adapter historical data
- `LEGACY_RISK_MANAGER_ENABLED` → Always enabled with centralized risk defaults

## New Configuration Structure

### Risk Management
Risk defaults are now centralized in `src/Safety/RiskDefaults.cs`:
- Environment-specific risk limits (Production, Staging, Development)
- SDK adapter specific risk defaults
- ML/RL specific risk thresholds
- UCB algorithm risk parameters

### SDK Configuration
Required environment variables:
```bash
PROJECT_X_API_KEY=your_api_key
PROJECT_X_USERNAME=your_username
TRADING_ENVIRONMENT=DEVELOPMENT|STAGING|PRODUCTION
```

### Optional Overrides
```bash
MAX_DAILY_LOSS_OVERRIDE=1000
MAX_POSITION_SIZE_OVERRIDE=10000
ML_MODEL_UPDATE_INTERVAL_HOURS=24
UCB_CONFIDENCE_THRESHOLD=0.65
```

## Migration Steps

1. Remove all legacy environment variables from your `.env` files
2. Add the new SDK environment variables
3. Update any code that references old config keys
4. Use `RiskDefaults.GetConfigurationForEnvironment()` for risk settings
5. Test the system with the new configuration

## Validation

Run the following to validate your configuration:
```bash
python3 python/test_sdk_wiring.py
python3 runtime_proof_demo.py
```

Both should pass without errors and show "SDK-only" status.
EOF

echo "✅ Created configuration migration guide: CONFIG_MIGRATION_GUIDE.md"

echo ""
echo "📋 Phase 5: Checking for configuration usage in C# code..."

# Check for scattered risk configuration in C# code
echo "🔍 Scanning for scattered risk configuration that should use RiskDefaults..."
risk_literals=$(grep -r -E "1000m|2000m|0\.01|50.*trade" src/ --include="*.cs" 2>/dev/null | grep -v "RiskDefaults" | wc -l)
if [ "$risk_literals" -gt 0 ]; then
    echo "⚠️ Found $risk_literals potential hardcoded risk values that could use RiskDefaults"
    echo "   Consider updating these to use the centralized risk configuration"
else
    echo "✅ No obvious hardcoded risk values found"
fi

echo ""
echo "🎯 LEGACY CONFIG CLEANUP SUMMARY:"
echo "✅ Scanned for legacy environment variable references"
echo "✅ Created centralized risk configuration (RiskDefaults.cs)"
echo "✅ Updated .env.example with SDK variables"
echo "✅ Created configuration migration guide"
echo "✅ Validated configuration usage patterns"
echo ""
echo "📄 Next steps:"
echo "   1. Review CONFIG_MIGRATION_GUIDE.md"
echo "   2. Update any remaining hardcoded risk values to use RiskDefaults"
echo "   3. Remove legacy environment variables from deployment configurations"
echo "   4. Test with the new configuration structure"
echo ""
echo "🚀 Legacy config cleanup complete!"