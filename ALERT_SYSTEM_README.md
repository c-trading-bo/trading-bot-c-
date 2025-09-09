# Trading Bot Alert System

## Overview

Production-grade monitoring and alerting system for model health, latency, and deployment events. Supports both email and Slack notifications with configurable thresholds.

## Features

### ✅ Alert Service
- **Email Alerts**: SMTP-based email notifications
- **Slack Alerts**: Rich webhook-based Slack messages with color coding and severity indicators
- **Critical Alerts**: Automatic escalation to both email and Slack for urgent issues
- **Configurable Severity Levels**: Info, Warning, Error, Critical

### ✅ Model Health Monitoring
- **Confidence Drift Detection**: Tracks deviation from baseline model confidence
- **Brier Score Monitoring**: Monitors prediction accuracy and calibration  
- **Feature Drift Detection**: Identifies changes in input feature distributions
- **Feature Parity Checks**: Validates expected vs. actual feature values

### ✅ Latency Monitoring
- **Decision Latency**: Tracks time for decision processing
- **Order Latency**: Monitors order execution time
- **Consecutive Threshold Breaches**: Alerts only after N consecutive violations
- **Statistical Analysis**: P50, P95, P99 percentile calculations

### ✅ Deployment Monitoring
- **Production Promotions**: Alerts when models are promoted to production
- **Canary Rollouts**: Notifications for canary deployment starts and failures
- **Automatic Rollbacks**: Critical alerts when rollbacks are triggered
- **Deployment Health Status**: Overall deployment system health tracking

## Configuration

Add the following environment variables to your `.env` or `.env.local` file:

```bash
# Email Configuration
ALERT_EMAIL_SMTP=smtp.gmail.com:587
ALERT_EMAIL_FROM=your-email@example.com
ALERT_EMAIL_TO=alerts@your-company.com

# Slack Configuration  
ALERT_SLACK_WEBHOOK=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Alert Thresholds
ALERT_CONFIDENCE_DRIFT_THRESHOLD=0.15
ALERT_BRIER_SCORE_THRESHOLD=0.3
ALERT_DECISION_LATENCY_THRESHOLD_MS=5000
ALERT_ORDER_LATENCY_THRESHOLD_MS=2000
ALERT_CONSECUTIVE_THRESHOLD_COUNT=3
```

## Usage

### Testing Alerts

Use the built-in test CLI to verify your alert configuration:

```bash
# Test all alert types
make test-alert

# Test specific alert types
make test-alert-email
make test-alert-slack

# Manual testing
./test-alert.sh all
./test-alert.sh email
./test-alert.sh slack
```

### Integrating into Your Code

```csharp
// Dependency Injection Setup
services.AddHttpClient();
services.AddSingleton<IAlertService, AlertService>();
services.AddSingleton<IModelHealthMonitor, ModelHealthMonitor>();
services.AddSingleton<ILatencyMonitor, LatencyMonitor>();
services.AddSingleton<IModelDeploymentManager, ModelDeploymentManager>();

// Model Health Monitoring
var modelHealthMonitor = serviceProvider.GetService<IModelHealthMonitor>();
modelHealthMonitor.RecordPrediction(confidence: 0.85, actualOutcome: 1.0, features);

// Latency Monitoring
var latencyMonitor = serviceProvider.GetService<ILatencyMonitor>();
using var tracker = latencyMonitor.StartDecisionTracking("Strategy evaluation");
// ... do work ...
// tracker automatically records latency when disposed

// Manual latency recording
latencyMonitor.RecordDecisionLatency(1250.0, "Manual measurement");

// Deployment Management
var deploymentManager = serviceProvider.GetService<IModelDeploymentManager>();
await deploymentManager.PromoteModelToProductionAsync("MyModel", "v2.1.0");
await deploymentManager.StartCanaryRolloutAsync("MyModel", "v2.2.0", trafficPercentage: 0.1);

// Direct Alert Service Usage
var alertService = serviceProvider.GetService<IAlertService>();
await alertService.SendCriticalAlertAsync("System Failure", "Critical component is down");
await alertService.SendModelHealthAlertAsync("MyModel", "High drift detected", metrics);
```

## Alert Types

### Model Health Alerts
- **Confidence Drift**: When model confidence significantly deviates from baseline
- **High Brier Score**: When prediction accuracy falls below threshold
- **Feature Drift**: When input features change significantly from historical patterns
- **Feature Parity Failures**: When features don't match expected values

### Latency Alerts  
- **High Decision Latency**: When decision processing takes too long
- **High Order Latency**: When order execution is slow
- **Consecutive Violations**: Only alerts after multiple consecutive threshold breaches

### Deployment Alerts
- **Production Promotions**: Success/failure notifications for production deployments
- **Canary Rollouts**: Start/failure notifications for canary deployments  
- **Rollbacks**: Critical alerts when models are rolled back
- **Deployment Health**: Overall deployment system status

## Architecture

```
src/
├── Infrastructure/Alerts/
│   ├── AlertService.cs          # Core alert service with email/Slack
│   └── Alerts.csproj           # Project file
├── Monitoring/
│   ├── ModelHealthMonitor.cs    # Model health tracking
│   ├── LatencyMonitor.cs       # Latency measurement and alerting
│   ├── ModelDeploymentManager.cs # Deployment event management
│   └── Monitoring.csproj       # Project file
tools/AlertTestCli/             # CLI testing application
tests/Integration/              # Integration tests for alert system
```

## Testing

The alert system includes comprehensive testing:

1. **CLI Test Application**: `tools/AlertTestCli/` - Manual testing of all alert types
2. **Integration Tests**: `tests/Integration/AlertServiceTests.cs` and `MonitoringAlertTests.cs`
3. **Make Commands**: Simple `make test-alert` commands for quick validation

## Examples

### Email Alert Example
```
Subject: [WARNING] Model Health Alert: MyModel_v1.0
Body:
Timestamp: 2024-09-09 21:30:00 UTC

Model: MyModel_v1.0
Issue: Confidence drift detected
Metrics:
{
  "ConfidenceDrift": 0.25,
  "BrierScore": 0.35,
  "PredictionCount": 150
}
```

### Slack Alert Example  
```
⚠️ Trading Bot Alert

Severity: Warning
Timestamp: 2024-09-09 21:30:00 UTC
Message: Model Health Alert
Model: MyModel_v1.0
Issue: Confidence drift detected
```

## Requirements

- .NET 8.0+
- SMTP server access (for email alerts)
- Slack webhook URL (for Slack alerts)
- Environment variables configured

## Status

✅ **COMPLETE** - All requirements implemented and tested:
- [x] Alert Service with email and Slack support
- [x] Model health monitoring with drift detection
- [x] Latency monitoring with configurable thresholds
- [x] Deployment event alerting
- [x] Integration tests and CLI testing tools
- [x] Make command for easy testing (`make test-alert`)
- [x] No TODO comments or placeholder code
- [x] Functional alerts that load configuration from .env