# Intelligence Infrastructure

This directory contains the complete Intelligence Infrastructure for the trading bot, providing cloud-based market analysis and signal generation that runs completely separate from the C# trading bot.

## Architecture Overview

The Intelligence Infrastructure follows a **separation of concerns** design:
- **Intelligence System**: Collects data, trains models, generates signals (Python + GitHub Actions)
- **Trading Bot**: Executes trades with optional intelligence consumption (C#)

## Directory Structure

```
Intelligence/
├── workflows/          # GitHub Actions workflows (not used - workflows in .github/workflows/)
├── scripts/           # Python data collection and analysis scripts
│   ├── collect_news.py          # GDELT news collector (runs every 5 min)
│   ├── collect_market_data.py   # Market data collector (runs after close)
│   ├── build_features.py        # Feature engineering (runs nightly)
│   ├── train_models.py          # ML model training (runs nightly)
│   ├── generate_signals.py      # Signal generation (runs before market open)
│   └── generate_daily_report.py # Daily analysis reports
├── data/
│   ├── raw/           # Raw collected data
│   │   ├── news/      # GDELT news data and sentiment analysis
│   │   ├── indices/   # SPX, VIX, market indices data
│   │   └── calendars/ # Economic calendar events
│   ├── features/      # ML-ready feature matrices
│   ├── signals/       # Generated trading signals for bot consumption
│   └── trades/        # Trade results for feedback loop
├── models/            # Trained ML models (.joblib files)
└── reports/           # Daily analysis reports (JSON + HTML)
```

## How It Works

### 1. Data Collection Pipeline
- **News Collection** (every 5 minutes during market hours): Collects GDELT news, analyzes sentiment
- **Market Data Collection** (daily after close): Collects SPX, VIX, indices, volatility metrics
- **Feature Engineering** (nightly): Combines news + market data into ML features

### 2. Model Training Pipeline
- **Model Training** (nightly): Trains RandomForest/LogisticRegression models on historical features
- **Signal Generation** (before market open): Uses trained models to generate daily trading signals

### 3. Bot Integration
- **IntelligenceService.cs**: Optional service that reads `signals/latest.json`
- **Graceful Degradation**: Bot continues normally if intelligence unavailable
- **Feedback Loop**: Bot logs trade results to `trades/results.jsonl` for model improvement

## GitHub Actions Workflows

The intelligence pipeline runs entirely on GitHub Actions:

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `news_pulse.yml` | Every 5 min (market hours) | Collect GDELT news, analyze sentiment |
| `market_data.yml` | Daily after close (4:30 PM ET) | Collect SPX/VIX/indices data |
| `ml_trainer.yml` | Nightly (2:00 AM ET) | Train ML models, build features |
| `daily_report.yml` | Pre-market (8:00 AM ET) | Generate signals, daily analysis |

## Signal Format

The intelligence system generates signals in this format for bot consumption:

```json
{
  "date": "2024-09-01",
  "regime": "Trending",
  "newsIntensity": 45.2,
  "isCpiDay": false,
  "isFomcDay": false,
  "modelConfidence": 0.73,
  "primaryBias": "Long",
  "setups": [
    {
      "timeWindow": "Opening30Min",
      "direction": "Long",
      "confidenceScore": 0.68,
      "suggestedRiskMultiple": 1.2,
      "rationale": "Trending market with positive news sentiment"
    }
  ],
  "generatedAt": "2024-09-01T13:00:00Z",
  "version": "1.0"
}
```

## Bot Integration

### Minimal Integration (Recommended Start)
```csharp
// In your bot startup
var intelligence = await _intelligenceService.GetLatestIntelligenceAsync();
if (intelligence != null)
{
    _logger.LogInformation($"[INTEL] Regime: {intelligence.Regime}, Bias: {intelligence.PrimaryBias}");
}
```

### Progressive Enhancement Options
1. **Pre-Market Consultation**: Log intelligence at startup
2. **Position Sizing**: Adjust position size based on confidence
3. **Trade Filtering**: Skip trades in unfavorable regimes

## Key Benefits

- ✅ **Complete Isolation**: Intelligence never directly trades
- ✅ **Graceful Degradation**: Bot works with or without intelligence
- ✅ **Cloud Learning**: 24/7 data collection and model training
- ✅ **Local Execution**: All trading remains local-only
- ✅ **Progressive Enhancement**: Add intelligence features gradually
- ✅ **Audit Trail**: Full logging of intelligence usage vs pure bot logic

## Getting Started

1. **Enable Workflows**: Intelligence pipelines run automatically via GitHub Actions
2. **Monitor Data**: Check `Intelligence/data/` directories for collected data
3. **Review Signals**: Check `Intelligence/data/signals/latest.json` for generated signals
4. **Optional Integration**: Add `IntelligenceService` to your bot when ready

## Configuration

Intelligence settings in `appsettings.json`:
```json
{
  "Intelligence": {
    "Enabled": false,          // Start with false for testing
    "SignalsPath": "../Intelligence/data/signals/latest.json",
    "MaxConfidenceThreshold": 0.7,
    "MinConfidenceThreshold": 0.3,
    "UseForSizing": false,     // Progressive feature flags
    "UseForFiltering": false,
    "UseForTiming": false
  }
}
```

## Dependencies

Python packages required for intelligence scripts:
- `requests` - HTTP requests for data collection
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning models
- `yfinance` - Yahoo Finance market data
- `joblib` - Model serialization

These are automatically installed by GitHub Actions workflows.