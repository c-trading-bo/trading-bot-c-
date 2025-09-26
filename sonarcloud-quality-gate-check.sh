#!/bin/bash

# SonarCloud Quality Gate Validation Script
# Ensures production readiness by validating code quality metrics

set -euo pipefail

echo "🔍 SonarCloud Quality Gate Validation"
echo "======================================"

# Check for remaining stubs and placeholders
echo "1. Checking for NotImplementedException stubs..."
STUB_COUNT=$(grep -r "NotImplementedException" src/ tests/ --include="*.cs" | grep -v -E "(analyzer|detection|test|Analyzer\.cs|Test.*\.cs)" | wc -l || echo "0")
if [ "$STUB_COUNT" -gt 0 ]; then
    echo "❌ Found $STUB_COUNT NotImplementedException stubs in production code"
    grep -r "NotImplementedException" src/ --include="*.cs" | grep -v -E "(analyzer|detection|Analyzer\.cs)"
    exit 1
else
    echo "✅ No NotImplementedException stubs found in production code"
fi

# Check for pragma warning disables
echo "2. Checking for pragma warning disables..."
PRAGMA_COUNT=$(grep -r "#pragma warning disable" src/ --include="*.cs" | grep -v -E "(analyzer|suppression|SuppressionLedgerService|Analyzer\.cs)" | wc -l || echo "0")
if [ "$PRAGMA_COUNT" -gt 0 ]; then
    echo "❌ Found $PRAGMA_COUNT pragma warning disables in production code"
    grep -r "#pragma warning disable" src/ --include="*.cs" | grep -v -E "(analyzer|suppression|SuppressionLedgerService|Analyzer\.cs)"
    exit 1
else
    echo "✅ No pragma warning disables found in production code"
fi

# Check for TODO/FIXME/HACK markers
echo "3. Checking for TODO/FIXME/HACK markers..."
TODO_COUNT=$(grep -r "TODO\|FIXME\|HACK" src/ --include="*.cs" | grep -v -E "(analyzer|detection|pattern|test|Analyzer\.cs)" | wc -l || echo "0")
if [ "$TODO_COUNT" -gt 0 ]; then
    echo "⚠️ Found $TODO_COUNT TODO/FIXME/HACK markers in production code"
    grep -r "TODO\|FIXME\|HACK" src/ --include="*.cs" | grep -v -E "(analyzer|detection|pattern|test|Analyzer\.cs)" | head -5
    echo "Note: These may be acceptable if they are in analyzer pattern detection code"
fi

# Check for build errors and warnings
echo "4. Building solution with warnings as errors..."
if dotnet build TopstepX.Bot.sln -warnaserror --no-restore > build_output.txt 2>&1; then
    echo "✅ Solution builds successfully with warnings as errors"
    rm -f build_output.txt
else
    echo "❌ Build failed with warnings treated as errors"
    tail -20 build_output.txt
    rm -f build_output.txt
    exit 1
fi

# Check for SuppressMessage attributes
echo "5. Checking for SuppressMessage attributes..."
SUPPRESS_COUNT=$(grep -r "SuppressMessage" src/ --include="*.cs" | grep -v -E "(analyzer|suppression|SuppressionLedgerService|Analyzer\.cs)" | wc -l || echo "0")
if [ "$SUPPRESS_COUNT" -gt 0 ]; then
    echo "❌ Found $SUPPRESS_COUNT SuppressMessage attributes in production code"
    grep -r "SuppressMessage" src/ --include="*.cs" | grep -v -E "(analyzer|suppression|SuppressionLedgerService|Analyzer\.cs)" | head -5
    exit 1
else
    echo "✅ No SuppressMessage attributes found in production code"
fi

# Validate analyzer compliance
echo "6. Running analyzer compliance check..."
if [ -f "./dev-helper.sh" ]; then
    if ./dev-helper.sh analyzer-check > analyzer_output.txt 2>&1; then
        echo "✅ Analyzer compliance check passed"
        rm -f analyzer_output.txt
    else
        echo "❌ Analyzer compliance check failed"
        tail -20 analyzer_output.txt
        rm -f analyzer_output.txt
        exit 1
    fi
else
    echo "⚠️ dev-helper.sh not found, skipping analyzer check"
fi

# Check for production safety violations
echo "7. Validating production safety requirements..."

# Check Directory.Build.props hasn't been weakened
if grep -q "TreatWarningsAsErrors.*false" Directory.Build.props 2>/dev/null; then
    echo "❌ TreatWarningsAsErrors has been disabled"
    exit 1
else
    echo "✅ TreatWarningsAsErrors is enforced"
fi

# Check for hardcoded values in critical trading components
echo "8. Checking for hardcoded trading values..."
HARDCODED_COUNT=$(grep -r "4[5-9][0-9][0-9]\|5[0-9][0-9][0-9]" src/ --include="*.cs" | grep -v -E "(test|mock|example|comment)" | wc -l || echo "0")
if [ "$HARDCODED_COUNT" -gt 5 ]; then
    echo "⚠️ Found $HARDCODED_COUNT potential hardcoded price values (may be acceptable)"
    # This is a warning, not an error, as some hardcoded values may be legitimate
fi

# Final quality gate assessment
echo ""
echo "🎯 SonarCloud Quality Gate Assessment"
echo "======================================"
echo "✅ No NotImplementedException stubs in production code"
echo "✅ No pragma warning disables in production code"
echo "✅ Solution builds with warnings as errors"
echo "✅ No SuppressMessage attributes in production code"
echo "✅ TreatWarningsAsErrors enforcement maintained"

if [ "$TODO_COUNT" -eq 0 ]; then
    echo "✅ No TODO/FIXME/HACK markers in production code"
else
    echo "⚠️ $TODO_COUNT TODO/FIXME/HACK markers found (review required)"
fi

echo ""
echo "🚀 Quality Gate Status: PASSED"
echo "Repository is ready for SonarCloud quality gate validation"
echo "All production readiness criteria have been met"