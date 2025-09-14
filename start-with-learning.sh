#!/bin/bash

# Concurrent Learning Startup Script
# This script ensures your bot runs with concurrent historical learning enabled

echo "🤖 Starting Trading Bot with Concurrent Historical Learning..."

# Load concurrent learning configuration
if [ -f ".env.concurrent-learning" ]; then
    echo "📊 Loading concurrent learning configuration..."
    export $(cat .env.concurrent-learning | xargs)
else
    echo "⚠️  No .env.concurrent-learning file found, using defaults..."
    # Set default values for concurrent learning
    export CONCURRENT_LEARNING=1
    export RUN_LEARNING=1
    export BACKTEST_MODE=1
    export MAX_CONCURRENT_OPERATIONS=2
    export LEARNING_PRIORITY=LOW
    export LIVE_TRADING_PRIORITY=HIGH
    export CONCURRENT_LEARNING_INTERVAL_MINUTES=60
    export OFFLINE_LEARNING_INTERVAL_MINUTES=15
    export CONCURRENT_LEARNING_DAYS=7
    export OFFLINE_LEARNING_DAYS=30
fi

# Load main environment configuration
if [ -f ".env" ]; then
    echo "🔧 Loading main environment configuration..."
    export $(cat .env | xargs)
fi

# Display learning configuration
echo ""
echo "📈 CONCURRENT LEARNING CONFIGURATION:"
echo "   CONCURRENT_LEARNING: $CONCURRENT_LEARNING"
echo "   RUN_LEARNING: $RUN_LEARNING"
echo "   BACKTEST_MODE: $BACKTEST_MODE"
echo "   Learning during market hours: Every $CONCURRENT_LEARNING_INTERVAL_MINUTES minutes"
echo "   Learning during market closed: Every $OFFLINE_LEARNING_INTERVAL_MINUTES minutes"
echo ""

# Start the bot
echo "🚀 Starting Unified Trading Orchestrator with concurrent learning..."
cd src/UnifiedOrchestrator
dotnet run

echo "🛑 Trading Bot with concurrent learning stopped"