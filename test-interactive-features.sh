#!/bin/bash
# Quick validation script for interactive testing features

set -e

echo "🧪 Testing Interactive Testing Features"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0
total_tests=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    total_tests=$((total_tests + 1))
    echo -e "${BLUE}Test $total_tests: $test_name${NC}"
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        fail_count=$((fail_count + 1))
    fi
    echo ""
}

# Test 1: Risk calculation function
run_test "Risk Calculation Function" \
    "timeout 30 dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build -- --test-function risk-calc"

# Test 2: Tick rounding function
run_test "Tick Rounding Function" \
    "timeout 30 dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build -- --test-function tick-round"

# Test 3: Order evidence function
run_test "Order Evidence Function" \
    "timeout 30 dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build -- --test-function order-proof"

# Test 4: Market data function
run_test "Market Data Function" \
    "timeout 30 dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build -- --test-function market-data"

# Test 5: Strategy function
run_test "Strategy Function (S6)" \
    "timeout 30 dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build -- --test-function strategy S6"

# Test 6: dev-helper.sh integration
run_test "dev-helper.sh run-interactive command exists" \
    "grep -q 'cmd_run_interactive' dev-helper.sh"

# Test 7: dev-helper.sh test-function command exists
run_test "dev-helper.sh test-function command exists" \
    "grep -q 'cmd_test_function' dev-helper.sh"

# Test 8: Documentation exists
run_test "INTERACTIVE_TESTING_GUIDE.md exists" \
    "test -f INTERACTIVE_TESTING_GUIDE.md"

# Test 9: InteractiveTestingService exists
run_test "InteractiveTestingService.cs exists" \
    "test -f src/UnifiedOrchestrator/Services/InteractiveTestingService.cs"

# Test 10: Program.cs has interactive mode support
run_test "Program.cs supports --interactive flag" \
    "grep -q 'RunInteractiveModeAsync' src/UnifiedOrchestrator/Program.cs"

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Total Tests:  $total_tests"
echo -e "${GREEN}Passed:       $pass_count${NC}"
echo -e "${RED}Failed:       $fail_count${NC}"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed!${NC}"
    echo ""
    echo "Interactive testing features are ready to use:"
    echo "  ./dev-helper.sh run-interactive              - Step-by-step debugging"
    echo "  ./dev-helper.sh test-function risk-calc      - Test risk calculation"
    echo "  ./dev-helper.sh test-function tick-round     - Test price rounding"
    echo "  ./dev-helper.sh test-function order-proof    - Test order evidence"
    echo "  ./dev-helper.sh test-function strategy S6    - Test strategy"
    echo "  ./dev-helper.sh test-function market-data    - Test market data"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
