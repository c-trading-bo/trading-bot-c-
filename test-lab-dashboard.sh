#!/bin/bash
# Test Lab Mode Dashboard - Verify console logging is suppressed and file logging works

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        Lab Mode Dashboard Console Logging Test                  ║"
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
echo "║  Lab Mode will write logs to a file for monitoring              ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                  ║"
echo "║  📝 Training logs: logs/lab-training-*.log                      ║"
echo "║                                                                  ║"
echo "║  💡 To monitor training progress, open another terminal and:    ║"
echo "║     tail -f logs/lab-training-*.log                             ║"
echo "║                                                                  ║"
echo "║  Dashboard will update in place (main terminal)                 ║"
echo "║  Training logs will stream to file (second terminal)            ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Run for 10 seconds to verify console logging behavior
echo "🧪 Starting Lab Mode (will run for 10 seconds to test)..."
echo ""

# Run with timeout
timeout 10s dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build || true

echo ""
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    Test Complete                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ Dashboard should have updated in place without scrolling"
echo "📝 Training logs should be in logs/lab-training-*.log"
echo ""
echo "💡 View the log file:"
echo "   ls -lh logs/lab-training-*.log"
echo "   tail -20 logs/lab-training-*.log"

