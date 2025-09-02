# Local HTTPS Dashboard Setup

This project now uses a **local HTTPS dashboard** instead of cloud-based dashboards. All cloud dashboard functionality has been removed and replaced with a secure local HTTPS server.

## Quick Start

### 1. Generate SSL Certificate (Required)

Before starting the dashboard, you need to generate an SSL certificate for HTTPS:

**Option A: Using the provided script (Linux/macOS/Windows with bash)**
```bash
./generate-ssl-cert.sh
```

**Option B: Using PowerShell (Windows/Linux/macOS)**
```powershell
./generate-ssl-cert.ps1
```

**Option C: Manual generation with OpenSSL**
```bash
mkdir -p certs
openssl req -x509 -newkey rsa:4096 -keyout certs/localhost.key -out certs/localhost.crt -days 365 -nodes -subj "/C=US/ST=CA/L=SF/O=TradingBot/OU=Dev/CN=localhost"
```

### 2. Start the HTTPS Dashboard

```bash
cd src/StandaloneDashboard
dotnet run
```

### 3. Access the Dashboard

Open your browser and navigate to:
```
https://localhost:5050/dashboard
```

**⚠️ Security Warning**: Your browser will show a warning about the self-signed certificate. Click "Advanced" and "Proceed to localhost" to continue.

## Features

- 🔒 **Secure HTTPS**: All communications encrypted with SSL
- 🎮 **Bot Control**: Start, stop, and manage your trading bot
- 📊 **Real-time Monitoring**: Live data from your local bot
- 🧠 **Learning Status**: Monitor ML/RL training progress  
- ⚡ **System Health**: Connection status and performance metrics
- 🔄 **Live Updates**: Real-time data streaming via Server-Sent Events

## Configuration

### Environment Variables (.env.local)

```bash
# HTTPS enabled by default
ASPNETCORE_URLS=https://localhost:5050

# Bot API URL (also HTTPS)
BOT_API_URL=https://localhost:5000
```

### Launch Script

Use the enhanced launcher with HTTPS support:

```bash
# Paper trading (default)
./launch-bot-enhanced.ps1 -Mode paper

# Live trading  
./launch-bot-enhanced.ps1 -Mode live

# Custom port
./launch-bot-enhanced.ps1 -Mode paper -Port 5051
```

## Removed Cloud Components

The following cloud dashboard components have been **completely removed**:

- ❌ GitHub Pages integration (`index.html`)
- ❌ GitHub Actions dashboard workflows
- ❌ Cloud dashboard URLs and links
- ❌ GitHub dashboard registration code
- ❌ External cloud dashboard dependencies

## Security Notes

### Self-Signed Certificate

The generated certificate is self-signed for local development. This means:

- ✅ **Secure**: All traffic is encrypted with SSL/TLS
- ⚠️ **Browser Warning**: Your browser will show a security warning
- 🔒 **Local Only**: Certificate is only valid for localhost

### Production Use

For production deployment, replace the self-signed certificate with a proper SSL certificate from a Certificate Authority (CA).

## Troubleshooting

### Certificate Issues

If you get SSL certificate errors:

1. **Regenerate the certificate**:
   ```bash
   rm -rf certs/
   ./generate-ssl-cert.sh
   ```

2. **Check certificate files exist**:
   ```bash
   ls -la certs/
   # Should show: localhost.crt and localhost.key
   ```

### Browser Issues

If the dashboard doesn't load:

1. **Clear browser cache** for localhost
2. **Accept the security warning** by clicking "Advanced" → "Proceed to localhost"  
3. **Try incognito/private mode**

### Connection Issues

If the dashboard can't connect to the bot:

1. **Check bot is running** on the correct port
2. **Verify HTTPS URLs** in environment configuration
3. **Check firewall settings** for local HTTPS traffic

## Development

### Building

```bash
cd src/StandaloneDashboard
dotnet build
```

### Running in Development

```bash
cd src/StandaloneDashboard  
dotnet run --environment Development
```

### Logs

The dashboard logs all activity to the console and to the in-browser log stream. Monitor the console output for connection status and errors.

## Architecture

```
┌─────────────────┐    HTTPS    ┌─────────────────┐
│   Browser       │◄──────────►│ Dashboard       │
│ localhost:5050  │             │ (ASP.NET Core)  │
└─────────────────┘             └─────────────────┘
                                         │ HTTPS
                                         ▼
                                ┌─────────────────┐
                                │ Trading Bot     │
                                │ (OrchestratorAgent)│
                                └─────────────────┘
```

All communication is encrypted and runs locally - no cloud dependencies!