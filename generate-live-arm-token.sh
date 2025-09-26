#!/bin/bash

# Live Trading Arm Token Generator
# Creates a short-lived token for manual live trading authorization
# Usage: ./generate-live-arm-token.sh [duration_minutes]

set -euo pipefail

DURATION_MINUTES=${1:-60}  # Default 1 hour
ARM_FILE="state/live_arm.json"
BACKUP_FILE="state/live_arm_backup.json"

echo "🔐 Live Trading Arm Token Generator"
echo "====================================="

# Ensure state directory exists
mkdir -p state

# Generate a cryptographically secure random token
if command -v openssl &> /dev/null; then
    TOKEN=$(openssl rand -hex 32)
elif command -v head &> /dev/null && [ -r /dev/urandom ]; then
    TOKEN=$(head -c 32 /dev/urandom | xxd -p -c 32)
else
    echo "❌ Error: Cannot generate secure random token (openssl or /dev/urandom required)"
    exit 1
fi

# Calculate expiration time
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS date command
    EXPIRES_AT=$(date -u -v +${DURATION_MINUTES}M +"%Y-%m-%dT%H:%M:%SZ")
else
    # GNU date command (Linux)
    EXPIRES_AT=$(date -u -d "+${DURATION_MINUTES} minutes" +"%Y-%m-%dT%H:%M:%SZ")
fi

# Backup existing token if it exists
if [ -f "$ARM_FILE" ]; then
    cp "$ARM_FILE" "$BACKUP_FILE"
    echo "📋 Backed up existing token to $BACKUP_FILE"
fi

# Create the arm token file
cat > "$ARM_FILE" << EOF
{
  "token": "$TOKEN",
  "expires_at": "$EXPIRES_AT",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "duration_minutes": $DURATION_MINUTES,
  "created_by": "${USER:-unknown}",
  "hostname": "${HOSTNAME:-$(hostname 2>/dev/null || echo unknown)}"
}
EOF

echo "✅ Live arm token created: $ARM_FILE"
echo "🕐 Token expires at: $EXPIRES_AT (${DURATION_MINUTES} minutes)"
echo ""
echo "🔑 To enable live trading, set this environment variable:"
echo ""
echo "export LIVE_ARM_TOKEN=\"$TOKEN\""
echo ""
echo "📋 Or add to your .env file:"
echo "LIVE_ARM_TOKEN=$TOKEN"
echo ""
echo "⚠️  SECURITY NOTICE:"
echo "   - This token expires in $DURATION_MINUTES minutes"
echo "   - Delete $ARM_FILE when live trading session ends"  
echo "   - Never commit this token to version control"
echo "   - Token is only valid on this local machine"
echo ""
echo "🔒 To disable live trading immediately:"
echo "   rm $ARM_FILE"
echo "   # or create kill switch:"
echo "   touch state/kill.txt"

# Set secure file permissions
chmod 600 "$ARM_FILE"

echo ""
echo "✅ Live arm token ready. Start your trading bot now."