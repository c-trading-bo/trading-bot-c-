#!/bin/bash

# UCB API Smoke Tests
# Run these to verify the API is working correctly

API_BASE="http://localhost:5000"

echo "🧪 UCB API Smoke Tests"
echo "======================"
echo "📡 Testing API at: $API_BASE"
echo ""

# Test 1: Health check
echo "Test 1: Health Check"
echo "--------------------"
curl -s "$API_BASE/health" | python -m json.tool
echo ""

# Test 2: Get recommendation
echo "Test 2: Get Trading Recommendation"
echo "----------------------------------"
curl -s "$API_BASE/ucb/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "es_price": 5330.25,
    "nq_price": 19150.5,
    "es_volume": 120000,
    "nq_volume": 98000,
    "es_atr": 10.2,
    "nq_atr": 25.3,
    "vix": 14.1,
    "tick": 350,
    "add": 1200,
    "correlation": 0.82,
    "rsi_es": 56,
    "rsi_nq": 61,
    "instrument": "ES"
  }' | python -m json.tool
echo ""

# Test 3: Update P&L
echo "Test 3: Update Strategy P&L"
echo "---------------------------"
curl -s "$API_BASE/ucb/update_pnl" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "S2_mean_reversion",
    "pnl": 250.0
  }' | python -m json.tool
echo ""

# Test 4: Get stats after update
echo "Test 4: Get Updated Stats"
echo "------------------------"
curl -s "$API_BASE/ucb/stats" | python -m json.tool
echo ""

# Test 5: Update with loss
echo "Test 5: Update with Loss"
echo "-----------------------"
curl -s "$API_BASE/ucb/update_pnl" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "S3_compression_breakout", 
    "pnl": -150.0
  }' | python -m json.tool
echo ""

# Test 6: Get recommendation after P&L updates
echo "Test 6: Get Recommendation After Learning"
echo "----------------------------------------"
curl -s "$API_BASE/ucb/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "es_price": 5325.75,
    "nq_price": 19140.0,
    "es_volume": 115000,
    "nq_volume": 95000,
    "es_atr": 9.8,
    "nq_atr": 24.7,
    "vix": 15.2,
    "tick": -200,
    "add": 800,
    "correlation": 0.78,
    "rsi_es": 48,
    "rsi_nq": 52,
    "instrument": "NQ"
  }' | python -m json.tool
echo ""

# Test 7: Reset daily (for next trading day)
echo "Test 7: Reset Daily Stats"
echo "------------------------"
curl -s -X POST "$API_BASE/ucb/reset_daily" | python -m json.tool
echo ""

# Test 8: Verify reset
echo "Test 8: Verify Daily Reset"
echo "-------------------------"
curl -s "$API_BASE/ucb/stats" | python -m json.tool
echo ""

echo "✅ Smoke tests completed!"
echo ""
echo "Expected results:"
echo "- Health check: status='healthy'"
echo "- Recommendations: trade=true/false with strategy, confidence, position_size"
echo "- P&L updates: status='ok'"
echo "- Stats: daily_pnl and strategy_stats should reflect updates"
echo "- Reset: daily_pnl=0, current_drawdown=0"
