#!/bin/bash
# Test script for backtesting mode without API

# Set environment variables for backtest mode
export BACKTEST_MODE=1
export ENABLE_BACKTEST_UI=1
export SKIP_MODE_PROMPT=1
export DRY_RUN=1

# Optional: Set backtest parameters
export BACKTEST_SYMBOL=ES
export BACKTEST_MODEL=CVaR-PPO
export BACKTEST_DAYS=1

echo "🎬 Starting Backtest Mode Test..."
echo "   Mode: Backtest with UI"
echo "   Symbol: $BACKTEST_SYMBOL"
echo "   Days: $BACKTEST_DAYS"
echo "   UI: Enabled"
echo ""

cd /home/runner/work/QBot/QBot/src/UnifiedOrchestrator
timeout 30s dotnet run || echo "Test completed or timed out"
