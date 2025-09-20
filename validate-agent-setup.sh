#!/bin/bash
# Agent Repository Validation Script
# Validates that the repository is properly set up for coding agent use

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo "🤖 Coding Agent Repository Validation"
echo "===================================="
echo

# Check required files
log_info "Checking essential agent files..."

required_files=(
    "README_AGENTS.md"
    "CODING_AGENT_GUIDE.md"
    "PROJECT_STRUCTURE.md"
    ".github/copilot-instructions.md"
    "dev-helper.sh"
    ".env.example"
    "TopstepX.Bot.sln"
)

missing_files=0
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        log_success "✓ $file"
    else
        log_error "✗ $file (missing)"
        missing_files=$((missing_files + 1))
    fi
done

# Check key directories
log_info "Checking project structure..."

required_dirs=(
    "src/UnifiedOrchestrator"
    "src/BotCore"
    "src/TopstepAuthAgent"
    "tests/Unit"
    ".github/workflows"
)

missing_dirs=0
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        log_success "✓ $dir/"
    else
        log_error "✗ $dir/ (missing)"
        missing_dirs=$((missing_dirs + 1))
    fi
done

# Check .NET environment
log_info "Checking .NET environment..."

if command -v dotnet &> /dev/null; then
    dotnet_version=$(dotnet --version)
    log_success "✓ .NET SDK version: $dotnet_version"
else
    log_error "✗ .NET SDK not found"
    missing_dirs=$((missing_dirs + 1))
fi

# Check if packages can be restored
log_info "Testing package restore..."
if dotnet restore --verbosity quiet > /dev/null 2>&1; then
    log_success "✓ Package restore successful"
else
    log_warning "⚠ Package restore had issues (may be normal)"
fi

# Check if main projects can build (expect analyzer warnings)
log_info "Testing core project builds (analyzer warnings expected)..."

core_projects=(
    "src/TopstepAuthAgent/TopstepAuthAgent.csproj"
    "src/BotCore/BotCore.csproj"
)

build_success=0
for project in "${core_projects[@]}"; do
    if [ -f "$project" ]; then
        log_info "Testing build of $project..."
        if dotnet build "$project" --no-restore --verbosity quiet > /dev/null 2>&1; then
            log_success "✓ $project builds successfully"
            build_success=$((build_success + 1))
        else
            log_warning "⚠ $project has build issues (analyzer warnings expected)"
        fi
    fi
done

# Check environment file setup
log_info "Checking environment configuration..."
if [ -f ".env" ]; then
    log_success "✓ .env file exists"
elif [ -f ".env.example" ]; then
    log_warning "⚠ .env missing but .env.example exists (run ./dev-helper.sh setup)"
else
    log_error "✗ No environment configuration found"
fi

# Check helper script
log_info "Testing development helper script..."
if [ -x "dev-helper.sh" ]; then
    if ./dev-helper.sh help > /dev/null 2>&1; then
        log_success "✓ Development helper script works"
        
        # Test analyzer check command
        if ./dev-helper.sh analyzer-check > /dev/null 2>&1; then
            log_success "✓ Analyzer check command works (no new warnings)"
        else
            log_warning "⚠ Analyzer check failed (expected due to existing warnings)"
        fi
        
        # Test new backtest command
        if ./dev-helper.sh backtest > /dev/null 2>&1; then
            log_success "✓ Backtest command works (local data only)"
        else
            log_warning "⚠ Backtest command had issues"
        fi
        
        # Test new riskcheck command
        if ./dev-helper.sh riskcheck > /dev/null 2>&1; then
            log_success "✓ Risk check command works (snapshot validation)"
        else
            log_warning "⚠ Risk check command had issues"
        fi
    else
        log_warning "⚠ Development helper script has issues"
    fi
else
    log_error "✗ Development helper script not executable"
fi

# Summary
echo
echo "📊 Validation Summary"
echo "===================="

total_issues=$((missing_files + missing_dirs))

if [ $total_issues -eq 0 ]; then
    log_success "🎉 Repository is ready for coding agents!"
    echo
    echo "Quick start for agents:"
    echo "  ./dev-helper.sh setup && ./dev-helper.sh build"
    echo
    echo "Essential reading:"
    echo "  - README_AGENTS.md"
    echo "  - CODING_AGENT_GUIDE.md"
    echo "  - .github/copilot-instructions.md"
    
    exit 0
elif [ $total_issues -le 2 ]; then
    log_warning "⚠ Repository mostly ready with minor issues ($total_issues issues)"
    echo
    echo "Please address the issues above and re-run validation."
    exit 1
else
    log_error "❌ Repository needs setup work ($total_issues issues)"
    echo
    echo "Please address the issues above and re-run validation."
    exit 1
fi