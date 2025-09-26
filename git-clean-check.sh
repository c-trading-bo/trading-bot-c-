#!/bin/bash

# Generated artifact sanity check before commits
# Prevents regenerable build debris from being committed to Git

set -euo pipefail

echo "🔍 Git Clean Check - Validating no build debris will be committed"
echo "================================================================="

# Check for staging files that should be ignored
VIOLATIONS_FOUND=false

# Check for current_* and build_* files
if git diff --cached --name-only | grep -E '^(current_|build_)' >/dev/null 2>&1; then
    echo "❌ Found current_* or build_* files staged:"
    git diff --cached --name-only | grep -E '^(current_|build_)'
    VIOLATIONS_FOUND=true
fi

# Check for SARIF files
if git diff --cached --name-only | grep '\.sarif$' >/dev/null 2>&1; then
    echo "❌ Found .sarif files staged:"
    git diff --cached --name-only | grep '\.sarif$'
    VIOLATIONS_FOUND=true
fi

# Check for CRITICAL_ALERT files
if git diff --cached --name-only | grep '^CRITICAL_ALERT_' >/dev/null 2>&1; then
    echo "❌ Found CRITICAL_ALERT_* files staged:"
    git diff --cached --name-only | grep '^CRITICAL_ALERT_'
    VIOLATIONS_FOUND=true
fi

# Check for analysis/violations/progress files
if git diff --cached --name-only | grep -E '\.(txt|log)$' | grep -E '(analysis|violations|progress|build)' >/dev/null 2>&1; then
    echo "❌ Found analysis/violations/progress files staged:"
    git diff --cached --name-only | grep -E '\.(txt|log)$' | grep -E '(analysis|violations|progress|build)'
    VIOLATIONS_FOUND=true
fi

# Check for bin/obj directories
if git diff --cached --name-only | grep -E '/(bin|obj)/' >/dev/null 2>&1; then
    echo "❌ Found bin/obj directory contents staged:"
    git diff --cached --name-only | grep -E '/(bin|obj)/'
    VIOLATIONS_FOUND=true
fi

# Check for .idea/.checkpoints directories
if git diff --cached --name-only | grep -E '/(\\.idea|\\.checkpoints)/' >/dev/null 2>&1; then
    echo "❌ Found .idea/.checkpoints directory contents staged:"
    git diff --cached --name-only | grep -E '/(\\.idea|\\.checkpoints)/'
    VIOLATIONS_FOUND=true
fi

if [ "$VIOLATIONS_FOUND" = true ]; then
    echo ""
    echo "🚫 COMMIT BLOCKED - Regenerable artifacts detected!"
    echo ""
    echo "These files should not be committed as they are generated on demand:"
    echo "• current_* and build_* files (build debris)"
    echo "• *.sarif files (analyzer reports)" 
    echo "• CRITICAL_ALERT_* files (temporary alerts)"
    echo "• *analysis*.txt, *violations*.txt, *progress*.txt (reports)"
    echo "• bin/, obj/ directories (build outputs)"
    echo "• .idea/, .checkpoints/ directories (IDE/checkpoint data)"
    echo ""
    echo "To fix:"
    echo "  git reset HEAD <files>     # Unstage the problematic files"
    echo "  git clean -fd              # Remove untracked generated files"
    echo "  ./git-clean-check.sh       # Verify clean state"
    echo ""
    exit 1
fi

echo "✅ All staged files are appropriate for commit"
echo "✅ No regenerable build artifacts detected"
echo ""
echo "Safe to commit!"