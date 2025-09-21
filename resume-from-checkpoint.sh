#!/bin/bash

# Simple Checkpoint Resume Script
# Quick resume command for crash recovery

set -e

echo "🔄 RESUMING ANALYZER CLEANUP FROM LAST CHECKPOINT"
echo "================================================="

# Check if checkpoint executor exists
if [ ! -f "checkpoint-executor.sh" ]; then
    echo "❌ checkpoint-executor.sh not found"
    exit 1
fi

# Try to resume
if ./checkpoint-executor.sh resume; then
    echo "✅ Resume completed successfully"
else
    echo "❌ Resume failed - checking status"
    ./checkpoint-executor.sh status
    echo ""
    echo "💡 Manual recovery options:"
    echo "   ./checkpoint-executor.sh start     - Start fresh"
    echo "   ./checkpoint-executor.sh continue  - Continue current checkpoint"
    echo "   ./checkpoint-executor.sh status    - Show detailed status"
fi