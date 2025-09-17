#!/bin/bash

# Setup script for Production Enforcement Pre-Commit Hooks
# This script configures Git to use the custom production enforcement hooks

echo "🔧 Setting up Production Enforcement Pre-Commit Hooks..."

# Configure Git to use our custom hooks directory
git config core.hooksPath .githooks

# Make sure the pre-commit hook is executable
chmod +x .githooks/pre-commit

echo "✅ Production enforcement hooks are now active!"
echo ""
echo "📋 What this means:"
echo "   • All commits will be scanned for non-production patterns"
echo "   • Hardcoded business values will block commits"
echo "   • Mock/fake/stub/placeholder code is not allowed"
echo "   • TODO/FIXME/HACK comments will prevent commits"
echo "   • All violations must be fixed with configuration-driven code"
echo ""
echo "🚀 To test the hooks, run: git commit -m 'test commit'"
echo "   (This should fail if there are any violations in your changes)"