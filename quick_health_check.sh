#!/bin/bash
# Quick health check for ES/NQ system

echo "🔍 ES/NQ System Health Check"
echo "=========================="

# Check data files
echo "📊 Data Sources:"
if [ -f "Intelligence/data/options/es_nq_flow.json" ]; then
    echo "  ✅ Options flow data available"
else
    echo "  ❌ Options flow data missing"
fi

if [ -f "Intelligence/data/news/es_nq_sentiment.json" ]; then
    echo "  ✅ News sentiment data available"
else
    echo "  ❌ News sentiment data missing"
fi

if [ -f "Intelligence/data/regime/es_nq_regime.json" ]; then
    echo "  ✅ Regime detection data available"
else
    echo "  ❌ Regime detection data missing"
fi

# Check workflows
echo ""
echo "⚙️  Workflows:"
if [ -f ".github/workflows/es_nq_news_sentiment.yml" ]; then
    echo "  ✅ News sentiment workflow configured"
else
    echo "  ❌ News sentiment workflow missing"
fi

# Run dashboard
echo ""
echo "📈 System Dashboard:"
python3 monitor_es_nq.py
