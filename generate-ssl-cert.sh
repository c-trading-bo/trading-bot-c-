#!/bin/bash

# Generate SSL Certificate for Local HTTPS Dashboard
# This script creates a self-signed certificate for localhost development

echo "🔒 Generating SSL Certificate for Local HTTPS Dashboard"
echo "================================================"

# Create certs directory if it doesn't exist
mkdir -p certs

# Check if openssl is available
if ! command -v openssl &> /dev/null; then
    echo "❌ Error: openssl is not installed"
    echo "Please install openssl first:"
    echo "  Ubuntu/Debian: sudo apt-get install openssl"
    echo "  macOS: brew install openssl"
    echo "  Windows: Download from https://www.openssl.org/"
    exit 1
fi

# Generate private key and certificate
echo "🔑 Generating private key..."
openssl genrsa -out certs/localhost.key 4096

echo "📜 Generating certificate..."
openssl req -new -x509 -key certs/localhost.key -out certs/localhost.crt -days 365 \
    -subj "/C=US/ST=CA/L=SF/O=TradingBot/OU=Dev/CN=localhost" \
    -extensions v3_req \
    -config <(cat <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req

[req_distinguished_name]

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = 127.0.0.1
IP.1 = 127.0.0.1
EOF
)

# Set proper permissions
chmod 600 certs/localhost.key
chmod 644 certs/localhost.crt

echo ""
echo "✅ SSL Certificate generated successfully!"
echo ""
echo "Files created:"
echo "  📄 certs/localhost.crt (Certificate)"
echo "  🔑 certs/localhost.key (Private Key)"
echo ""
echo "🚀 You can now start the HTTPS dashboard:"
echo "  cd src/StandaloneDashboard"
echo "  dotnet run"
echo ""
echo "🌐 Access your dashboard at:"
echo "  https://localhost:5050/dashboard"
echo ""
echo "⚠️  Note: Your browser will show a security warning"
echo "   because this is a self-signed certificate."
echo "   Click 'Advanced' and 'Proceed to localhost' to continue."
echo ""