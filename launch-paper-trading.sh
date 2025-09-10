#!/bin/bash

echo "🚀 TopstepX Paper Trading Bot Launcher"
echo "======================================="
echo "This will start your bot in paper trading mode with live TopstepX endpoints"
echo "but using simulated contracts until full API access is enabled."
echo ""

cd /workspaces/trading-bot-c-

# Check if bot is already running
if pgrep -f "dotnet.*UnifiedOrchestrator" > /dev/null; then
    echo "🛑 Bot is already running. Stopping first..."
    pkill -f "dotnet.*UnifiedOrchestrator"
    sleep 2
fi

# Set environment for paper trading
export ENABLE_DRY_RUN="false"
export ENABLE_AUTO_EXECUTION="false" 
export ENABLE_PAPER_TRADING="true"
export TOPSTEPX_USE_SIMULATED_CONTRACTS="true"

echo "🎯 Configuration:"
echo "   📊 Paper Trading: ENABLED"
echo "   🔗 TopstepX Endpoints: LIVE (api.topstepx.com)"
echo "   📋 Contracts: SIMULATED (until full API access)"
echo "   🛡️ Safety: DRY_RUN disabled, AUTO_EXECUTION disabled"
echo ""

echo "🚀 Starting Unified Trading Orchestrator..."
echo "   Press Ctrl+C to stop"
echo ""

# Start the bot
dotnet run --project src/UnifiedOrchestrator
