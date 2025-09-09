# Trading Bot Alert System Makefile

.PHONY: test-alert test-alert-email test-alert-slack test-alert-all help

# Default target
help:
	@echo "Trading Bot Alert System"
	@echo ""
	@echo "Available targets:"
	@echo "  test-alert          - Test all alert types (email, Slack, monitoring)"
	@echo "  test-alert-email    - Test email alerts only"
	@echo "  test-alert-slack    - Test Slack alerts only"
	@echo "  test-alert-all      - Test all alert types (same as test-alert)"
	@echo ""
	@echo "Configuration:"
	@echo "  Ensure .env or .env.local contains:"
	@echo "    ALERT_EMAIL_SMTP=your-smtp-server"
	@echo "    ALERT_EMAIL_FROM=your-email@example.com"
	@echo "    ALERT_EMAIL_TO=alerts@your-company.com"
	@echo "    ALERT_SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

# Test all alert types
test-alert:
	@echo "Testing all alert types..."
	@./test-alert.sh all

# Test email alerts only
test-alert-email:
	@echo "Testing email alerts..."
	@./test-alert.sh email

# Test Slack alerts only
test-alert-slack:
	@echo "Testing Slack alerts..."
	@./test-alert.sh slack

# Test all alert types (alias)
test-alert-all: test-alert

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