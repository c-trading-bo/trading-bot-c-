#!/bin/bash
# Auto-launch Lab Mode training with proper menu selection
# This script automatically selects:
#   Option 2: Lab Mode (Historical Training)
#   Option 2: Manual Training (Run Now)

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║            QBot Lab Mode - Automatic Launch                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will start Lab Mode training immediately (bypassing Sunday schedule)"
echo ""

# Set environment variables for Lab Mode
export ASPNETCORE_ENVIRONMENT=Lab
export LAB_MODE=1
export FORCE_LAB_NOW=1

echo "[*] Building project..."
dotnet build src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -c Release --verbosity quiet

if [ $? -ne 0 ]; then
    echo "[✗] Build failed"
    exit 1
fi

echo "[✓] Build successful"
echo ""
echo "[*] Launching Lab Mode..."
echo "    Auto-selecting: [2] Lab Mode → [2] Manual Training"
echo ""

# Create input file with menu selections (newlines for Linux)
echo -e "2\n2" > /tmp/lab_input.txt

# Launch with input redirection
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build -c Release < /tmp/lab_input.txt

# Cleanup
rm -f /tmp/lab_input.txt
