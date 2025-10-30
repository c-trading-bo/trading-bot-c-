#!/bin/bash
# Run backtest mode completely offline using saved historical bars
# This script demonstrates that the bot can execute trades on historical data without any API connection

echo "╔════════════════════════════════════════════════════════════════════════════════╗"
echo "║           OFFLINE BACKTEST MODE - No API Required                             ║"
echo "╚════════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This script runs the bot in backtest mode using ONLY saved historical bars."
echo "No API connection is required - the bot runs completely offline."
echo ""

# Set environment variables for offline backtest mode
export BACKTEST_MODE=1
export SKIP_MODE_PROMPT=1
export DRY_RUN=1

# Disable UI for clearer logging output
export ENABLE_BACKTEST_UI=0

# Set backtest parameters
export BACKTEST_SYMBOL=ES
export BACKTEST_MODEL=CVaR-PPO
export BACKTEST_DAYS=1

# Use backtest-specific configuration
export ASPNETCORE_ENVIRONMENT=backtest

echo "Configuration:"
echo "  Symbol: $BACKTEST_SYMBOL"
echo "  Model: $BACKTEST_MODEL"
echo "  Days back: $BACKTEST_DAYS"
echo "  Data source: datasets/quotes/es_quotes.json (local file)"
echo "  UI: Disabled (for clearer logging)"
echo ""
echo "Expected output:"
echo "  ✅ Data loading from local file"
echo "  ✅ Historical bars being processed"
echo "  ✅ Trading decisions being made"
echo "  ✅ Trades being executed (simulated)"
echo "  ✅ Final P&L summary"
echo ""
echo "Starting backtest..."
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

cd src/UnifiedOrchestrator
dotnet run --configuration Release

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Backtest completed!"
echo ""
echo "Check the output above to verify:"
echo "  1. Historical data was loaded from local files (not API)"
echo "  2. The bot processed historical bars"
echo "  3. Trading decisions were made"
echo "  4. Trades were executed on the historical data"
echo "  5. Final P&L summary was shown"
echo ""
