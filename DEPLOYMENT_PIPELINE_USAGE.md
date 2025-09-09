# TopStep Trading Bot - Deployment Pipeline Usage Guide

This guide shows how to use the comprehensive deployment pipeline that automatically handles TopStep credentials and ensures production readiness.

## 🚀 Quick Start

### 1. Set Environment Variables for TopStep Credentials

The bot automatically detects credentials from multiple environment variable patterns:

```bash
# Primary pattern (recommended)
export TOPSTEPX_USERNAME="your_topstep_username"
export TOPSTEPX_API_KEY="your_topstep_api_key"
export TOPSTEPX_ACCOUNT_ID="your_account_id"

# Alternative patterns also supported:
# TOPSTEP_USERNAME, TSX_USERNAME, TRADING_USERNAME, BOT_USERNAME, LIVE_USERNAME
# TOPSTEP_API_KEY, TSX_API_KEY, TRADING_API_KEY, BOT_API_KEY, LIVE_API_KEY
```

### 2. Run the Complete Deployment Pipeline

```bash
cd src/Infrastructure.TopstepX
dotnet run
```

### 3. Understand Exit Codes

The system uses specific exit codes to indicate different states:
- `0` = Success (production ready)
- `2` = Credential issues  
- `4` = Test failures
- `8` = Production gate failure
- `9` = Critical system error

## 📊 Pipeline Phases

### Phase 1: Automatic Credential Detection
- Scans 12+ environment variable patterns
- Checks secure file storage
- Provides detailed source reporting

### Phase 2: Staging Environment Setup
- Configures environment to match TopStep production
- Sets conservative risk limits
- Validates connectivity to TopStep services

### Phase 3: Comprehensive Testing (22 tests across 7 categories)
- **Infrastructure**: Environment, file system, network, JSON
- **Credentials**: Discovery, patterns, staging setup
- **Integration**: TopStep API, SignalR hubs, JWT validation
- **Components**: Strategy loading, risk engine, market data
- **End-to-End**: Full bot simulation, signal processing, risk integration
- **Performance**: Latency, throughput, memory usage
- **Safety**: Kill switch, risk limits, critical systems

### Phase 4: Comprehensive Reporting
- Health score calculation (target: 90%+)
- Performance metrics (latency/throughput)
- Security compliance analysis
- Technical debt tracking
- Production readiness assessment

### Phase 5: Auto-Remediation
- Automatically fixes environment issues
- Sets up missing configurations
- Flags items requiring manual review

### Phase 6: Production Gate (6 gates)
1. **Preflight**: Environment health, critical systems, variables, network, security
2. **Testing**: 95%+ test pass rate required
3. **Performance**: Latency, throughput, resource usage, stability
4. **Security**: Credentials, network, configuration, compliance
5. **Auto-Remediation**: Issue resolution success
6. **Final Assessment**: Overall readiness (85%+ required)

## 🔧 CI/CD Integration

### GitHub Actions Setup

1. **Set Repository Secrets:**
   ```
   TOPSTEPX_USERNAME = your_username
   TOPSTEPX_API_KEY = your_api_key  
   TOPSTEPX_ACCOUNT_ID = your_account_id
   ```

2. **Pipeline automatically runs on:**
   - Push to main/master branches
   - Pull requests
   - Release builds

3. **CI Results:**
   - Artifacts saved for 30 days
   - Detailed reports in `reports/` directory
   - Exit codes determine CI status

## 📈 Results Interpretation

### With Valid Credentials:
- ✅ **Credential Detection**: SUCCESS
- ✅ **Staging Deployment**: SUCCESS  
- ⚠️ **Test Suite**: 86.4% pass rate (needs 95% for production)
- ✅ **Reporting**: 89.2% health score
- ✅ **Auto-Remediation**: 1+ issues fixed
- ⚠️ **Production Gate**: 60% readiness (needs 85% for production)

### Without Credentials:
- ❌ **Credential Detection**: FAILED
- ❌ **Staging Deployment**: FAILED
- ❌ **Test Suite**: 72.7% pass rate
- ✅ **Reporting**: 79.6% health score
- ✅ **Auto-Remediation**: Limited success
- ❌ **Production Gate**: 40% readiness

## 📊 Generated Reports

### Summary Report (`summary_report_*.txt`)
```
=== COMPREHENSIVE SYSTEM REPORT ===
Overall Health Score: 89.2 %
Pass Rate: 86.4 %
Security Score: 93.3%
Production Ready: ❌ NO
```

### Pipeline Report (`pipeline_execution_report_*.json`)
```json
{
  "Summary": {
    "CredentialDetection": { "Success": true, "Source": "Environment" },
    "TestSuite": { "PassRate": "86.4 %" },
    "HealthScore": "89.2 %"
  }
}
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Credential Detection Failed (Exit Code 2)**
   - Set `TOPSTEPX_USERNAME` and `TOPSTEPX_API_KEY` environment variables
   - Check spelling and ensure no extra spaces
   - Try alternative patterns like `TOPSTEP_USERNAME`

2. **Test Failures (Exit Code 4)**
   - Review test results in reports
   - Common failures: network connectivity, external dependencies
   - Tests require ~95% pass rate for production

3. **Production Gate Blocked (Exit Code 8)**
   - Need 85%+ readiness score
   - Address security issues
   - Improve test pass rate
   - Resolve manual review items

## 🔒 Security Considerations

- Credentials are never logged or exposed
- HTTPS-only communication enforced
- Conservative risk limits in staging
- File permissions set appropriately
- Environment variable masking in logs

## 🎯 Production Deployment Checklist

- [ ] Valid TopStep credentials configured
- [ ] Test pass rate ≥ 95%
- [ ] Health score ≥ 90%
- [ ] Security score ≥ 85%
- [ ] Readiness score ≥ 85%
- [ ] No critical manual review items
- [ ] All 6 production gates pass
- [ ] Exit code = 0

Once all criteria are met, the system is ready for production deployment!