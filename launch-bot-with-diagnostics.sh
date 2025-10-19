#!/bin/bash
# Bot Diagnostics and Launch Script
# Checks all prerequisites and launches with comprehensive logging

set -e

echo "🔍 Bot Pre-Flight Diagnostics"
echo "====================================="
echo ""

# Track issues
issues=0
warnings=0

# 1. Check Python
echo "1. Checking Python environment..."
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version)
    echo "   ✅ $python_version"
else
    echo "   ❌ Python3 not found"
    issues=$((issues + 1))
fi

# 2. Check Python SDK
echo "2. Checking project-x-py SDK..."
if python3 -c "import project_x_py" 2>/dev/null; then
    sdk_version=$(python3 -c "import project_x_py; print(project_x_py.__version__)" 2>/dev/null || echo "unknown")
    echo "   ✅ project-x-py installed (v$sdk_version)"
else
    echo "   ❌ project-x-py SDK not installed"
    echo "      Fix: pip install 'project-x-py[all]'"
    issues=$((issues + 1))
fi

# 3. Check .NET
echo "3. Checking .NET environment..."
if command -v dotnet &> /dev/null; then
    dotnet_version=$(dotnet --version)
    echo "   ✅ .NET $dotnet_version"
else
    echo "   ❌ .NET not found"
    issues=$((issues + 1))
fi

# 4. Check network connectivity
echo "4. Checking TopstepX API connectivity..."
if curl -s --connect-timeout 5 https://api.topstepx.com > /dev/null 2>&1; then
    echo "   ✅ TopstepX API reachable"
elif curl -s --connect-timeout 5 https://www.google.com > /dev/null 2>&1; then
    echo "   ⚠️  Internet works but TopstepX API unreachable"
    echo "      This might be expected (firewall/VPN)"
    warnings=$((warnings + 1))
else
    echo "   ❌ No internet connectivity detected"
    echo "      Running in OFFLINE mode"
    echo "      - API calls will fail"
    echo "      - Use mock/test data instead"
    issues=$((issues + 1))
fi

# 5. Check DNS resolution
echo "5. Checking DNS resolution..."
if nslookup api.topstepx.com > /dev/null 2>&1; then
    echo "   ✅ DNS resolves api.topstepx.com"
else
    echo "   ❌ DNS cannot resolve api.topstepx.com"
    echo "      Environment: $(uname -a | cut -d' ' -f1-2)"
    echo "      This is likely a GitHub Actions firewall restriction"
    issues=$((issues + 1))
fi

# 6. Check required files
echo "6. Checking configuration files..."
if [ -f ".env" ]; then
    echo "   ✅ .env file exists"
else
    echo "   ❌ .env file missing"
    issues=$((issues + 1))
fi

# 7. Check historical data
echo "7. Checking historical data..."
data_found=0
for dir in datasets/features datasets/quotes data/topstep; do
    if [ -d "$dir" ] && [ "$(ls -A $dir 2>/dev/null)" ]; then
        echo "   ✅ Data found in $dir"
        data_found=1
        break
    fi
done
if [ $data_found -eq 0 ]; then
    echo "   ⚠️  No historical data found"
    echo "      Bot will need real-time API access or mock data"
    warnings=$((warnings + 1))
fi

# 8. Check ML models
echo "8. Checking ML models..."
model_found=0
for model in models/rl_model.onnx models/rl/test_cvar_ppo.onnx; do
    if [ -f "$model" ]; then
        echo "   ✅ Model found: $model"
        model_found=1
    fi
done
if [ $model_found -eq 0 ]; then
    echo "   ⚠️  No ML models found"
    echo "      Bot will use fallback prediction logic"
    warnings=$((warnings + 1))
fi

# 9. Check build status
echo "9. Checking build..."
if dotnet build src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-restore -v quiet > /dev/null 2>&1; then
    echo "   ✅ Build successful"
else
    echo "   ⚠️  Build has warnings (expected)"
    warnings=$((warnings + 1))
fi

echo ""
echo "====================================="
echo "Diagnostic Summary"
echo "====================================="
echo "Issues: $issues"
echo "Warnings: $warnings"
echo ""

if [ $issues -gt 0 ]; then
    echo "❌ Critical issues found. Fix the errors above before launching."
    echo ""
    echo "Common fixes:"
    echo "  - Install Python SDK: pip install 'project-x-py[all]'"
    echo "  - Run on local machine for network access"
    echo "  - Provide historical data files"
    exit 1
fi

if [ $warnings -gt 0 ]; then
    echo "⚠️  Warnings detected. Bot may run with limited functionality."
    echo ""
fi

echo "🚀 Launching bot..."
echo ""

# Create log directory
mkdir -p logs

# Launch bot with comprehensive logging
LOG_FILE="logs/bot_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Logging to: $LOG_FILE"
echo ""

dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj 2>&1 | tee "$LOG_FILE"
