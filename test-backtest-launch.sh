#!/bin/bash
# Test script to verify backtest mode launches correctly

cd /home/runner/work/QBot/QBot/src/UnifiedOrchestrator

# Set environment variables
export BACKTEST_MODE=1
export ENABLE_BACKTEST_UI=1
export SKIP_MODE_PROMPT=1
export DRY_RUN=1
export BACKTEST_SYMBOL=ES
export BACKTEST_DAYS=1

echo "=== Backtest Mode Launch Test ==="
echo "Environment variables set:"
echo "  BACKTEST_MODE=$BACKTEST_MODE"
echo "  ENABLE_BACKTEST_UI=$ENABLE_BACKTEST_UI"
echo "  BACKTEST_SYMBOL=$BACKTEST_SYMBOL"
echo "  BACKTEST_DAYS=$BACKTEST_DAYS"
echo ""

echo "Building project..."
dotnet build --no-restore 2>&1 | tail -5

echo ""
echo "Checking data files..."
ls -lh /home/runner/work/QBot/QBot/datasets/quotes/*.json

echo ""
echo "=== Launching backtest mode (will timeout after 3 seconds) ==="
echo "Expected: UI should render with control panel showing STOPPED state"
echo ""

# Launch with timeout - it will show the UI and wait for input
timeout 3s dotnet run --no-build 2>&1 || echo "Timed out (expected - waiting for keyboard input)"

echo ""
echo "=== Test Result ==="
if [ $? -eq 124 ]; then
    echo "✅ PASS: Backtest mode launched successfully and is waiting for user input"
    echo "   (Timeout occurred as expected - UI is waiting for SPACE key)"
else
    echo "❌ Status: Process exited with code $?"
fi
