@echo off
REM UCB API Smoke Tests for Windows
REM Run these to verify the API is working correctly

set API_BASE=http://localhost:5000

echo 🧪 UCB API Smoke Tests
echo ======================
echo 📡 Testing API at: %API_BASE%
echo.

REM Test 1: Health check
echo Test 1: Health Check
echo --------------------
curl -s "%API_BASE%/health"
echo.
echo.

REM Test 2: Get recommendation  
echo Test 2: Get Trading Recommendation
echo ----------------------------------
curl -s "%API_BASE%/ucb/recommend" ^
  -H "Content-Type: application/json" ^
  -d "{\"es_price\":5330.25,\"nq_price\":19150.5,\"es_volume\":120000,\"nq_volume\":98000,\"es_atr\":10.2,\"nq_atr\":25.3,\"vix\":14.1,\"tick\":350,\"add\":1200,\"correlation\":0.82,\"rsi_es\":56,\"rsi_nq\":61,\"instrument\":\"ES\"}"
echo.
echo.

REM Test 3: Update P&L
echo Test 3: Update Strategy P^&L
echo ---------------------------
curl -s "%API_BASE%/ucb/update_pnl" ^
  -H "Content-Type: application/json" ^
  -d "{\"strategy\":\"S2_mean_reversion\",\"pnl\":250.0}"
echo.
echo.

REM Test 4: Get stats
echo Test 4: Get Updated Stats
echo ------------------------
curl -s "%API_BASE%/ucb/stats"
echo.
echo.

REM Test 5: Reset daily
echo Test 5: Reset Daily Stats
echo ------------------------
curl -s -X POST "%API_BASE%/ucb/reset_daily"
echo.
echo.

echo ✅ Smoke tests completed!
echo.
echo Expected results:
echo - Health check: status='healthy'
echo - Recommendations: trade=true/false with strategy, confidence, position_size
echo - P^&L updates: status='ok'
echo - Stats: daily_pnl and strategy_stats should reflect updates
echo - Reset: daily_pnl=0, current_drawdown=0

pause
