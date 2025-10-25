#!/usr/bin/env bash
# Lab Mode Pre-Flight Check Script
# Validates that all required historical data files exist and are valid

# Don't exit on errors - we want to check all items
set +e

echo "========================================"
echo "Lab Mode Pre-Flight Validation"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
WARN=0
FAIL=0

# Check 1: Data directory exists
echo -n "Checking data directory... "
if [ -d "data/historical" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${RED}✗ FAIL${NC}"
    echo "  Directory 'data/historical' does not exist"
    echo "  Run: mkdir -p data/historical"
    ((FAIL++))
    exit 1
fi

# Check 2: Required 5m data files exist
echo ""
echo "Checking required 5-minute data files..."
for symbol in ES NQ; do
    FILE="data/historical/${symbol}_90days.json"
    echo -n "  ${symbol}_90days.json... "
    
    if [ -f "$FILE" ]; then
        SIZE=$(stat -f%z "$FILE" 2>/dev/null || stat -c%s "$FILE" 2>/dev/null || echo "0")
        if [ "$SIZE" -gt 102400 ]; then # > 100 KB
            # Check if file contains expected JSON structure
            if grep -q '"bars"' "$FILE" && grep -q '"timestamp"' "$FILE"; then
                echo -e "${GREEN}✓ PASS${NC} ($(numfmt --to=iec $SIZE 2>/dev/null || echo "${SIZE} bytes"))"
                ((PASS++))
            else
                echo -e "${RED}✗ FAIL${NC} (Invalid JSON structure)"
                ((FAIL++))
            fi
        else
            echo -e "${RED}✗ FAIL${NC} (File too small: ${SIZE} bytes)"
            echo "     Expected > 100 KB"
            ((FAIL++))
        fi
    else
        echo -e "${RED}✗ FAIL${NC} (File not found)"
        ((FAIL++))
    fi
done

# Check 3: Optional 1m data files
echo ""
echo "Checking optional 1-minute data files..."
for symbol in ES NQ; do
    FILE="data/historical/${symbol}_1m_90days.json"
    echo -n "  ${symbol}_1m_90days.json... "
    
    if [ -f "$FILE" ]; then
        SIZE=$(stat -f%z "$FILE" 2>/dev/null || stat -c%s "$FILE" 2>/dev/null || echo "0")
        if [ "$SIZE" -gt 102400 ]; then
            echo -e "${GREEN}✓ PRESENT${NC} ($(numfmt --to=iec $SIZE 2>/dev/null || echo "${SIZE} bytes"))"
            ((PASS++))
        else
            echo -e "${YELLOW}⚠ WARNING${NC} (File too small)"
            ((WARN++))
        fi
    else
        echo -e "${YELLOW}⚠ NOT FOUND${NC} (Training will use 5m data only)"
        ((WARN++))
    fi
done

# Check 4: Python executable
echo ""
echo -n "Checking Python availability... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${GREEN}✓ PASS${NC} ($PYTHON_VERSION)"
    ((PASS++))
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    echo -e "${GREEN}✓ PASS${NC} ($PYTHON_VERSION)"
    ((PASS++))
else
    echo -e "${YELLOW}⚠ WARNING${NC} (Python not found in PATH)"
    echo "  Python is needed to fetch historical data"
    echo "  Install: sudo apt-get install python3 (Linux)"
    echo "           brew install python3 (macOS)"
    ((WARN++))
fi

# Check 5: fetch-and-save-historical-data.py script
echo ""
echo -n "Checking data fetch script... "
if [ -f "fetch-and-save-historical-data.py" ]; then
    echo -e "${GREEN}✓ PASS${NC}"
    ((PASS++))
else
    echo -e "${YELLOW}⚠ WARNING${NC}"
    echo "  Script 'fetch-and-save-historical-data.py' not found"
    echo "  Data refresh will not be available"
    ((WARN++))
fi

# Summary
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo -e "${GREEN}Passed:  $PASS${NC}"
echo -e "${YELLOW}Warnings: $WARN${NC}"
echo -e "${RED}Failed:  $FAIL${NC}"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}✅ All critical checks passed!${NC}"
    echo -e "${GREEN}Lab Mode should be able to start training.${NC}"
    echo ""
    echo "To launch Lab Mode:"
    echo "  FORCE_LAB_NOW=1 dotnet run --project src/UnifiedOrchestrator"
    exit 0
else
    echo -e "${RED}❌ Some critical checks failed.${NC}"
    echo ""
    echo "To fix missing data files:"
    echo "  python3 fetch-and-save-historical-data.py"
    echo ""
    echo "For more help, see:"
    echo "  LAB_MODE_STARTUP_TROUBLESHOOTING.md"
    exit 1
fi
