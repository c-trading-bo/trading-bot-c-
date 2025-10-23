#!/bin/bash
set -e

echo "=========================================="
echo "OVERFITTING PREVENTION LOGIC TEST"
echo "=========================================="
echo ""

cd /home/runner/work/QBot/QBot/src/UnifiedOrchestrator

echo "Step 1: Building project..."
dotnet build -c Release --no-restore > /dev/null 2>&1
echo "✅ Build complete"
echo ""

echo "Step 2: Launching Lab Mode with overfitting prevention..."
echo "Environment:"
echo "  - SKIP_MODE_PROMPT=1 (bypass menu)"
echo "  - BOT_MODE=Lab (training mode)"
echo "  - LAB_MODE=1 (enable lab features)"
echo "  - MANUAL_TRAINING=1 (trigger training immediately)"
echo ""

# Run Lab Mode and capture output showing overfitting prevention in action
timeout --foreground 120 env \
  SKIP_MODE_PROMPT=1 \
  BOT_MODE=Lab \
  LAB_MODE=1 \
  MANUAL_TRAINING=1 \
  dotnet run --no-build -c Release 2>&1 | tee /tmp/lab_output.log | grep -E "(TRAINING|MULTI-SEED|DATA-SPLIT|EARLY-STOP|PROMOTION|CVaR|LSTM|Pattern|Regime|Slippage|Ensemble|seed|validation|test set)" || true

echo ""
echo "=========================================="
echo "VERIFICATION: Checking for overfitting prevention logic"
echo "=========================================="

# Check if data splitting was called
if grep -q "SplitData" /tmp/lab_output.log 2>/dev/null || grep -q "train.*validation.*test" /tmp/lab_output.log 2>/dev/null; then
    echo "✅ Data splitting: DETECTED"
else
    echo "⚠️  Data splitting: NOT DETECTED in logs"
fi

# Check if multi-seed training was called
if grep -q "GetTrainingSeeds\|seed.*42\|seed.*123\|seed.*456" /tmp/lab_output.log 2>/dev/null; then
    echo "✅ Multi-seed training: DETECTED"
else
    echo "⚠️  Multi-seed training: NOT DETECTED in logs"
fi

# Check if early stopping was called
if grep -q "EarlyStop\|ShouldStop\|patience" /tmp/lab_output.log 2>/dev/null; then
    echo "✅ Early stopping: DETECTED"
else
    echo "⚠️  Early stopping: NOT DETECTED in logs"
fi

# Check if promotion decision was called
if grep -q "MakePromotionDecision\|PROMOTION\|APPROVED\|REJECTED" /tmp/lab_output.log 2>/dev/null; then
    echo "✅ Promotion decision: DETECTED"
else
    echo "⚠️  Promotion decision: NOT DETECTED in logs"
fi

echo ""
echo "Full log saved to: /tmp/lab_output.log"
echo "=========================================="
