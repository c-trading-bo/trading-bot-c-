# Unified Environment Configuration & Gmail Alert Setup

## Overview

All bot credentials and configuration have been consolidated into a single `.env` file. This includes TopstepX authentication, GitHub API access, Gmail SMTP settings, and all alert thresholds.

## Quick Setup

1. **Copy the template:**
   ```bash
   cp .env.example .env
   ```

2. **Configure Gmail SMTP (for alerts):**
   - Generate a Gmail App Password: https://support.google.com/accounts/answer/185833
   - Update `.env` with your credentials:
   ```bash
   ALERT_EMAIL_USERNAME=kevinsuero072897@gmail.com
   ALERT_EMAIL_PASSWORD=your_16_character_app_password
   ```

3. **Test the setup:**
   ```bash
   make test-env
   make test-alert-email
   ```

4. **Launch the bot:**
   ```bash
   make run-orchestrator
   ```

## Environment File Structure

The consolidated `.env` file contains:

### 🔐 Authentication & API Access
- **TopstepX Credentials**: API key, username, JWT token
- **GitHub API**: Token and repository configuration
- **Gmail SMTP**: Complete email alert configuration

### ⚡ Trading Configuration
- **Trading Modes**: Paper trading, live connection settings
- **Risk Management**: Loss caps, position limits, slippage thresholds
- **Critical Systems**: Disaster recovery, correlation protection

### 📧 Alert System
- **Gmail SMTP**: Full authentication and TLS configuration
- **Slack Integration**: Webhook URL (optional)
- **Thresholds**: Model health, latency monitoring, deployment alerts

## Alert System Features

### Email Alerts (Gmail SMTP)
- ✅ Model health degradation alerts
- ✅ System latency warnings
- ✅ Deployment success/failure notifications
- ✅ Critical system alerts
- ✅ Secure TLS authentication

### Alert Types
1. **Model Health**: Confidence drift, Brier score monitoring
2. **Performance**: Decision latency, order execution timing
3. **Deployment**: Model promotion, canary rollouts, rollbacks
4. **Critical**: System failures, emergency situations

## Testing Commands

```bash
# Test environment loading
make test-env

# Test email alerts
make test-alert-email

# Test all alert types
make test-alert

# Test bot startup
make run-orchestrator
```

## Gmail App Password Setup

1. **Enable 2-Factor Authentication** on your Google account
2. **Generate App Password**:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate password for "Mail"
   - Copy the 16-character password

3. **Update .env file**:
   ```bash
   ALERT_EMAIL_PASSWORD=abcd efgh ijkl mnop
   ```

## Security Notes

⚠️ **IMPORTANT**: The `.env` file contains sensitive credentials and is excluded from version control via `.gitignore`.

✅ **Protected**: TopstepX JWT tokens, Gmail passwords, GitHub tokens
✅ **Encrypted**: TLS encryption for all SMTP communications
✅ **Isolated**: Each environment variable is scoped appropriately

## File Changes Made

### ✅ Consolidated Files
- **Created**: Single `.env` with all credentials
- **Created**: `.env.example` template
- **Created**: `test-bot-setup.sh` verification script

### ❌ Removed Files
- ~~`.env.github`~~ (merged into `.env`)
- ~~`.env.test`~~ (merged into `.env`)
- ~~`.env.sample.local`~~ (replaced with `.env.example`)
- ~~`.github/copilot_mechanic/.env`~~ (merged into `.env`)

### 🔧 Updated Components
- **AlertService**: Enhanced SMTP authentication with username/password
- **Makefile**: Added environment testing and bot launch commands
- **Configuration**: All services now load from unified `.env` file

## Environment Variables Reference

### Core Trading
```bash
TOPSTEPX_API_KEY=your_api_key
TOPSTEPX_USERNAME=your_username
TOPSTEPX_JWT=your_jwt_token
PAPER_MODE=1
AUTH_ALLOW=1
```

### Gmail Alerts
```bash
ALERT_EMAIL_SMTP=smtp.gmail.com
ALERT_EMAIL_PORT=587
ALERT_EMAIL_USE_TLS=true
ALERT_EMAIL_USERNAME=your_email@gmail.com
ALERT_EMAIL_PASSWORD=your_app_password
ALERT_EMAIL_FROM=your_email@gmail.com
ALERT_EMAIL_TO=your_email@gmail.com
```

### Alert Thresholds
```bash
ALERT_CONFIDENCE_DRIFT_THRESHOLD=0.15
ALERT_BRIER_SCORE_THRESHOLD=0.3
ALERT_DECISION_LATENCY_THRESHOLD_MS=5000
ALERT_ORDER_LATENCY_THRESHOLD_MS=2000
```

## Troubleshooting

### "Email: False" in logs
- Check `ALERT_EMAIL_PASSWORD` is set in `.env`
- Verify Gmail App Password is correct (16 characters)
- Ensure 2FA is enabled on Google account

### "JWT token invalid"
- Refresh TopstepX JWT token
- Check `TOPSTEPX_API_KEY` and `TOPSTEPX_USERNAME`
- Verify account credentials

### Environment not loading
- Ensure `.env` file exists in project root
- Check file permissions (readable)
- Run `make test-env` to verify loading

## Production Usage

This consolidated environment setup is production-ready with:
- 🔒 Secure credential management
- 📧 Real-time Gmail alerting
- 🚨 Critical system monitoring
- 📊 Performance tracking
- 🔄 Deployment notifications

The alert system will send immediate Gmail notifications for any model degradation, system latency issues, or deployment events, ensuring rapid incident response.