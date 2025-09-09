#!/bin/bash

# Alert Test Script
# Provides make test-alert functionality
# Usage: ./test-alert.sh [alert-type]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
CLI_PROJECT="$PROJECT_ROOT/tools/AlertTestCli"

echo "=== Trading Bot Alert Test ==="
echo "Project root: $PROJECT_ROOT"
echo "CLI project: $CLI_PROJECT"

# Check if .env file exists and warn about configuration
if [ ! -f "$PROJECT_ROOT/.env" ] && [ ! -f "$PROJECT_ROOT/.env.local" ]; then
    echo ""
    echo "⚠️  WARNING: No .env or .env.local file found!"
    echo "   Create .env.local with your alert configuration:"
    echo "   ALERT_EMAIL_SMTP=your-smtp-server"
    echo "   ALERT_EMAIL_FROM=your-email@example.com" 
    echo "   ALERT_EMAIL_TO=alerts@your-company.com"
    echo "   ALERT_SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
    echo ""
fi

# Build and run the CLI test application
echo "Building alert test CLI..."
cd "$CLI_PROJECT"
dotnet build --configuration Release --verbosity quiet

if [ $? -ne 0 ]; then
    echo "❌ Failed to build alert test CLI"
    exit 1
fi

echo "Running alert tests..."
if [ $# -eq 0 ]; then
    # No arguments - run all tests
    dotnet run --configuration Release -- all
else
    # Run specific test type
    dotnet run --configuration Release -- "$1"
fi

echo ""
echo "✅ Alert test completed!"
echo ""
echo "If you configured email/Slack properly, you should have received notifications."
echo "Check your email inbox and Slack channels for test alerts."