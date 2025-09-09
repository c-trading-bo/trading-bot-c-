# Trading Bot Unified Environment & Alert System Makefile

.PHONY: test-alert test-alert-email test-alert-slack test-alert-all test-env help

# Default target
help:
	@echo "Trading Bot Unified Environment & Alert System"
	@echo ""
	@echo "Environment Management:"
	@echo "  test-env            - Test environment file loading and bot credentials"
	@echo "  test-bot-setup      - Complete setup verification and credential check"
	@echo ""
	@echo "Alert Testing:"
	@echo "  test-alert          - Test all alert types (email, Slack, monitoring)"
	@echo "  test-alert-email    - Test email alerts only"
	@echo "  test-alert-slack    - Test Slack alerts only"  
	@echo "  test-alert-all      - Test all alert types (same as test-alert)"
	@echo ""
	@echo "Bot Operations:"
	@echo "  run-bot             - Launch the trading bot with unified environment"
	@echo "  run-orchestrator    - Run the orchestrator agent"
	@echo ""
	@echo "Configuration:"
	@echo "  All credentials are now in .env file (Gmail SMTP, TopstepX, GitHub, etc.)"
	@echo "  Copy .env.example to .env and fill in your values"
	@echo ""
	@echo "Gmail SMTP Setup:"
	@echo "  ALERT_EMAIL_SMTP=smtp.gmail.com"
	@echo "  ALERT_EMAIL_USERNAME=kevinsuero072897@gmail.com"
	@echo "  ALERT_EMAIL_PASSWORD=your_gmail_app_password"

# Test unified environment loading
test-env:
	@echo "Testing unified environment configuration..."
	@./test-bot-setup.sh

# Alias for test-env
test-bot-setup: test-env

# Test all alert types
test-alert:
	@echo "Testing all alert types..."
	@./test-alert.sh all

# Test email alerts only
test-alert-email:
	@echo "Testing Gmail SMTP email alerts..."
	@./test-alert.sh email

# Test Slack alerts only
test-alert-slack:
	@echo "Testing Slack alerts..."
	@./test-alert.sh slack

# Test all alert types (alias)
test-alert-all: test-alert

# Run the trading bot
run-bot:
	@echo "Starting trading bot with unified environment..."
	@dotnet run --project app/TradingBot/TradingBot.csproj

# Run the orchestrator agent
run-orchestrator:
	@echo "Starting orchestrator agent..."
	@dotnet run --project src/OrchestratorAgent/OrchestratorAgent.csproj

# Build the alert CLI tool
build-alert-cli:
	@echo "Building alert test CLI..."
	@cd tools/AlertTestCli && dotnet build --configuration Release

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@cd tools/AlertTestCli && dotnet clean
	@find . -name "bin" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "obj" -type d -exec rm -rf {} + 2>/dev/null || true