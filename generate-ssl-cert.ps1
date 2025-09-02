#!/usr/bin/env pwsh

<#
.SYNOPSIS
    Generate SSL Certificate for Local HTTPS Dashboard

.DESCRIPTION
    Creates a self-signed certificate for localhost development
    Works on Windows, macOS, and Linux with PowerShell

.EXAMPLE
    .\generate-ssl-cert.ps1
#>

Write-Host "🔒 Generating SSL Certificate for Local HTTPS Dashboard" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Yellow

# Create certs directory if it doesn't exist
if (-not (Test-Path "certs")) {
    New-Item -ItemType Directory -Path "certs" | Out-Null
    Write-Host "📁 Created certs directory" -ForegroundColor Cyan
}

# Check if OpenSSL is available
$opensslPath = Get-Command openssl -ErrorAction SilentlyContinue

if (-not $opensslPath) {
    Write-Host "❌ Error: openssl is not installed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install openssl first:" -ForegroundColor Yellow
    Write-Host "  Windows: choco install openssl (or download from https://www.openssl.org/)" -ForegroundColor White
    Write-Host "  macOS: brew install openssl" -ForegroundColor White
    Write-Host "  Linux: sudo apt-get install openssl" -ForegroundColor White
    Write-Host ""
    Write-Host "Alternative: Use Windows built-in certificate tools:" -ForegroundColor Yellow
    Write-Host "  New-SelfSignedCertificate -DnsName localhost -CertStoreLocation cert:\LocalMachine\My" -ForegroundColor White
    exit 1
}

try {
    # Generate private key
    Write-Host "🔑 Generating private key..." -ForegroundColor Cyan
    & openssl genrsa -out certs/localhost.key 4096
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate private key"
    }

    # Create temporary config file for certificate
    $configContent = @"
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = CA
L = SF
O = TradingBot
OU = Dev
CN = localhost

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = 127.0.0.1
IP.1 = 127.0.0.1
"@

    $configFile = "certs/cert.conf"
    $configContent | Out-File -FilePath $configFile -Encoding utf8

    # Generate certificate
    Write-Host "📜 Generating certificate..." -ForegroundColor Cyan
    & openssl req -new -x509 -key certs/localhost.key -out certs/localhost.crt -days 365 -config $configFile -extensions v3_req
    
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to generate certificate"
    }

    # Clean up temporary config file
    Remove-Item $configFile -ErrorAction SilentlyContinue

    # Set proper permissions (Unix-like systems)
    if ($IsLinux -or $IsMacOS) {
        chmod 600 certs/localhost.key
        chmod 644 certs/localhost.crt
    }

    Write-Host ""
    Write-Host "✅ SSL Certificate generated successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Files created:" -ForegroundColor Yellow
    Write-Host "  📄 certs/localhost.crt (Certificate)" -ForegroundColor White
    Write-Host "  🔑 certs/localhost.key (Private Key)" -ForegroundColor White
    Write-Host ""
    Write-Host "🚀 You can now start the HTTPS dashboard:" -ForegroundColor Green
    Write-Host "  cd src/StandaloneDashboard" -ForegroundColor White
    Write-Host "  dotnet run" -ForegroundColor White
    Write-Host ""
    Write-Host "🌐 Access your dashboard at:" -ForegroundColor Green
    Write-Host "  https://localhost:5050/dashboard" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⚠️  Note: Your browser will show a security warning" -ForegroundColor Yellow
    Write-Host "   because this is a self-signed certificate." -ForegroundColor Yellow
    Write-Host "   Click 'Advanced' and 'Proceed to localhost' to continue." -ForegroundColor Yellow
    Write-Host ""

} catch {
    Write-Host "❌ Error generating certificate: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Alternative for Windows users:" -ForegroundColor Yellow
    Write-Host "Run as Administrator and use:" -ForegroundColor White
    Write-Host "  `$cert = New-SelfSignedCertificate -DnsName localhost -CertStoreLocation cert:\LocalMachine\My" -ForegroundColor White
    Write-Host "  Export-Certificate -Cert `$cert -FilePath certs\localhost.crt" -ForegroundColor White
    exit 1
}