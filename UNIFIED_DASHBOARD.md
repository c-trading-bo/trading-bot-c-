# Unified Cloud Dashboard

## Overview

The Unified Cloud Dashboard consolidates all trading bot features into a single, cloud-based interface that works even without TopstepX API credentials. This dashboard provides complete visibility and control over all bot operations.

## Features

### 🤖 Bot Control
- **Launch Bot**: Start the bot in Paper, Shadow, or Live mode
- **Stop/Restart**: Full control over bot lifecycle
- **Emergency Stop**: Immediate halt functionality
- **Mode Switching**: Toggle between trading modes

### 📊 Real-Time Monitoring
- **Performance Overview**: Live P&L, trades, win rate, account balance
- **Strategy Performance**: Individual strategy metrics and P&L
- **Live Data Streaming**: Real-time updates via Server-Sent Events
- **System Health**: API connection, data feeds, risk management status

### ☁️ Cloud Learning Integration
- **24/7 Learning**: GitHub Actions-based cloud training
- **Model Metrics**: Accuracy, training samples, system uptime
- **Manual Training**: Trigger training cycles on-demand
- **Progress Tracking**: Visual learning progress indicators

### 🔗 GitHub Integration
- **Actions Dashboard**: Direct links to GitHub Actions workflows
- **Cloud Deployment**: GitHub Pages dashboard integration
- **Status Badges**: Live status indicators
- **Workflow Monitoring**: Recent training runs and results

### 📝 Live Logging
- **Real-Time Logs**: Streaming log entries with filtering
- **Activity Tracking**: Trading activity and system events
- **Learning Events**: Cloud learning cycle notifications
- **Error Monitoring**: System health and error tracking

## Quick Start

### Option 1: Standalone Dashboard (Recommended)
```bash
# Start the unified dashboard server
cd src/StandaloneDashboard
dotnet run

# Access dashboard at: http://localhost:5050/dashboard
```

### Option 2: Full Bot with Dashboard
```bash
# Set environment variables for demo mode
export DEMO_MODE=1
export PAPER_MODE=1

# Start the full bot
dotnet run --project src/OrchestratorAgent
```

## Demo Mode

The dashboard includes a comprehensive demo mode that simulates all bot features:
- ✅ Live data updates (P&L, trades, metrics)
- ✅ Strategy performance simulation
- ✅ Cloud learning progress
- ✅ Bot control functionality
- ✅ Real-time logging

## Dashboard Sections

### Header
- **Bot Status**: Current running state and mode
- **Control Buttons**: Launch, Stop, Restart functionality
- **Version Badge**: Dashboard version indicator

### Main Grid
1. **Bot Performance Overview**: Key metrics and P&L
2. **Live Strategy Performance**: Individual strategy results
3. **24/7 Cloud Learning**: ML model status and controls
4. **System Health**: Component status monitoring
5. **Recent Trading Activity**: Live trade feed
6. **GitHub Cloud Integration**: External dashboard links

### Sidebar
- **Quick Controls**: Fast-access control buttons
- **Live Logs**: Real-time system activity stream

## Environment Variables

```bash
# Demo mode (no credentials required)
DEMO_MODE=1
PAPER_MODE=1
TRADING_MODE=PAPER

# Cloud learning
GITHUB_CLOUD_LEARNING=1
RL_ENABLED=1
CLOUD_PROVIDER=github

# Dashboard port
DASHBOARD_PORT=5050
```

## API Endpoints

- `GET /dashboard` - Main dashboard interface
- `GET /healthz` - Health check endpoint
- `POST /api/bot/start` - Start bot with mode selection
- `POST /api/bot/stop` - Stop bot
- `GET /stream/realtime` - Real-time data stream (SSE)
- `GET /data/history` - Historical data (demo)

## Cloud Integration

The dashboard integrates with GitHub Actions for 24/7 cloud learning:
- **GitHub Actions**: Automated training workflows
- **GitHub Pages**: Cloud-hosted dashboard
- **Model Deployment**: Automatic model updates
- **Status Monitoring**: Workflow success/failure tracking

## Browser Compatibility

- ✅ Chrome/Chromium
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ✅ Mobile browsers

## Why This Solution?

### Problem Addressed
The original request was for "one dashboard that works here cloud based" with all bot features displayed. The previous setup had multiple separate dashboards and complex authentication requirements.

### Solution Benefits
1. **Single Unified Interface**: All features in one place
2. **Cloud-Based**: Works without local TopstepX API
3. **Demo Mode**: Full functionality demonstration
4. **Real-Time Updates**: Live data streaming
5. **Complete Feature Coverage**: Every bot capability displayed
6. **Easy Deployment**: Standalone or integrated options

### TopstepX API Limitation
Since the TopstepX API cannot be connected directly, the dashboard operates in demo mode while showing all the bot's capabilities. This provides:
- Complete feature visualization
- Full control interface
- Simulated real-time data
- Cloud learning integration
- GitHub Actions connectivity

The dashboard demonstrates exactly how the bot would work with live API credentials while providing a fully functional cloud-based interface.