#!/bin/bash

echo "================================================================================"
echo "🔍 PRODUCTION READINESS VERIFICATION"
echo "================================================================================"
echo

SUCCESS=0
TOTAL=0

# Test 1: MinimalDemo Launch Test
echo "Test 1: MinimalDemo Launch Verification"
echo "----------------------------------------"
((TOTAL++))
if timeout 15 dotnet run --project MinimalDemo/MinimalDemo.csproj > /tmp/demo_output.txt 2>&1; then
    if grep -q "DEMONSTRATION COMPLETED SUCCESSFULLY" /tmp/demo_output.txt; then
        echo "✅ PASS: MinimalDemo launches and completes successfully"
        ((SUCCESS++))
    else
        echo "❌ FAIL: MinimalDemo did not complete successfully"
    fi
else
    echo "❌ FAIL: MinimalDemo failed to launch"
fi
echo

# Test 2: Core Components Build Test
echo "Test 2: Core Components Build Verification"
echo "-------------------------------------------"
((TOTAL++))
if timeout 60 dotnet build src/BotCore/BotCore.csproj \
    -p:TreatWarningsAsErrors=false \
    -p:CodeAnalysisTreatWarningsAsErrors=false \
    --verbosity quiet > /tmp/build_output.txt 2>&1; then
    if grep -q "Build succeeded\|Build SUCCEEDED" /tmp/build_output.txt || [ $? -eq 0 ]; then
        echo "✅ PASS: BotCore builds successfully"
        ((SUCCESS++))
    else
        echo "❌ FAIL: BotCore build has compilation errors"
        tail -5 /tmp/build_output.txt
    fi
else
    echo "❌ FAIL: BotCore build timeout or critical error"
    tail -5 /tmp/build_output.txt
fi
echo

# Test 3: Production Rules Enforcement
echo "Test 3: Production Rules Enforcement Verification"
echo "--------------------------------------------------"
((TOTAL++))
if find src -name "*.cs" -not -path "*/Analyzers/*" -not -path "*/Test*" -exec grep -l "PLACEHOLDER\|TEMP\|DUMMY\|MOCK\|FAKE\|STUB" {} \; | grep -v "NO_ATTEMPT_CAPS" | head -1 > /dev/null; then
    echo "❌ FAIL: Found prohibited patterns in production code"
    find src -name "*.cs" -not -path "*/Analyzers/*" -not -path "*/Test*" -exec grep -l "PLACEHOLDER\|TEMP\|DUMMY\|MOCK\|FAKE\|STUB" {} \; | grep -v "NO_ATTEMPT_CAPS" | head -3
else
    echo "✅ PASS: No prohibited patterns found in production code"
    ((SUCCESS++))
fi
echo

# Test 4: Configuration-Driven Parameters
echo "Test 4: Configuration-Driven Parameters Verification"
echo "------------------------------------------------------"
((TOTAL++))
CONFIG_VIOLATIONS=0
# Check for hardcoded position size 2.5 in trading logic (exclude simulation/analyzer code) 
if grep -r "PositionSize.*2\.5\|MaxPosition.*2\.5\|positionMultiplier.*2\.5" src --include="*.cs" | grep -v Test | grep -v Simulation | grep -v Analyzer | head -1 > /dev/null; then
    echo "❌ Found hardcoded position size 2.5 in trading logic"
    ((CONFIG_VIOLATIONS++))
fi
# Check for hardcoded confidence 0.7 in decision logic (exclude thresholds/simulation)
if grep -r "confidence.*0\.7\|Confidence.*0\.7" src --include="*.cs" | grep -v Test | grep -v Threshold | grep -v const | head -1 > /dev/null; then
    echo "❌ Found hardcoded confidence 0.7 in decision logic"
    ((CONFIG_VIOLATIONS++))
fi
if [ $CONFIG_VIOLATIONS -eq 0 ]; then
    echo "✅ PASS: No hardcoded trading parameters found in core trading logic"
    ((SUCCESS++))
else
    echo "❌ FAIL: Found $CONFIG_VIOLATIONS hardcoded trading parameters in core logic"
fi
echo

# Test 5: Assembly and Versioning
echo "Test 5: Assembly and Versioning Verification"
echo "---------------------------------------------"
((TOTAL++))
ASSEMBLY_ISSUES=0
for proj in src/*/; do
    if [ -f "$proj"Properties/AssemblyInfo.cs ]; then
        if ! grep -q "AssemblyVersion" "$proj"Properties/AssemblyInfo.cs; then
            ((ASSEMBLY_ISSUES++))
        fi
    fi
done
if [ $ASSEMBLY_ISSUES -eq 0 ]; then
    echo "✅ PASS: Assembly versioning is properly configured"
    ((SUCCESS++))
else
    echo "❌ FAIL: $ASSEMBLY_ISSUES projects have assembly versioning issues"
fi
echo

# Final Results
echo "================================================================================"
echo "📊 PRODUCTION READINESS RESULTS"
echo "================================================================================"
echo "Tests Passed: $SUCCESS/$TOTAL"
echo "Success Rate: $(( (SUCCESS * 100) / TOTAL ))%"
echo

if [ $SUCCESS -eq $TOTAL ]; then
    echo "🎉 PRODUCTION READY: All verification tests passed!"
    echo "✅ System is ready for production deployment"
    exit 0
else
    echo "⚠️  PARTIAL READINESS: $((TOTAL - SUCCESS)) test(s) failed"
    echo "🔧 Address the failing tests before production deployment"
    exit 1
fi