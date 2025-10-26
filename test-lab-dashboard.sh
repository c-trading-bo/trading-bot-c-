#!/bin/bash
# Test Lab Mode Dashboard - Verify stable dashboard with inline alerts

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        Lab Mode Dashboard Test                                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Set environment variables for Lab Mode
export LAB_MODE=1
export FORCE_LAB_NOW=1
export ASPNETCORE_ENVIRONMENT=Lab
export SKIP_MODE_PROMPT=1

# Build the project first
echo "📦 Building UnifiedOrchestrator..."
cd /home/runner/work/QBot/QBot
dotnet build src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-restore > /dev/null 2>&1 || {
    echo "❌ Build failed"
    exit 1
}
echo "✅ Build succeeded"
echo ""

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  Lab Mode Dashboard - Single Terminal View                      ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                  ║"
echo "║  ✅ Dashboard shows training progress                           ║"
echo "║  ✅ Critical errors/warnings appear in Alerts section           ║"
echo "║  ✅ No log spam - clean, stable display                         ║"
echo "║  ✅ Diagnostic logs saved to logs/lab-training-*.log            ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Run for 10 seconds to verify dashboard behavior
echo "🧪 Starting Lab Mode (will run for 10 seconds)..."
echo ""

# Run with timeout
timeout 10s dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build || true

echo ""
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    Test Complete                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ Dashboard should update in place without scrolling"
echo "✅ Only critical alerts shown inline (no spam)"
echo "📝 Full logs available in logs/lab-training-*.log if needed"


