#!/bin/bash
# Test Lab Mode Dashboard - Verify console logging is suppressed

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

# Run for 10 seconds to verify console logging behavior
echo "🧪 Starting Lab Mode (will run for 10 seconds to test console output)..."
echo "Expected: Dashboard updates in place (ANSI escape codes)"
echo "Not Expected: Scrolling log messages"
echo ""
echo "Press Ctrl+C after observing behavior..."
echo ""

# Run with timeout
timeout 10s dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build || true

echo ""
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    Test Complete                                 ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ If you saw a stable dashboard updating in place, the fix works!"
echo "❌ If you saw scrolling logs, the fix didn't work."
