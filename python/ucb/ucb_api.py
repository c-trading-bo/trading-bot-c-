from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import os
import sys
import math
import asyncio
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Make sure model is importable
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from neural_ucb_topstep import UCBIntegration

# ==============================================
# LIFESPAN MANAGEMENT WITH LOCK
# ==============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 UCB API starting...")
    
    # Initialize UCB
    persistence_path = os.getenv("UCB_PERSISTENCE_PATH", "ucb_state.pkl")
    weights_path = os.getenv("UCB_WEIGHTS_PATH", "neural_ucb_topstep.pth")
    app.state.ucb = UCBIntegration(weights_path=weights_path, persistence_path=persistence_path)
    
    # Create async lock for thread safety
    app.state.lock = asyncio.Lock()
    
    # Start state persistence task
    app.state.persistence_task = asyncio.create_task(
        state_persistence_loop(app.state.ucb, app.state.lock)
    )
    
    logger.info("✅ UCB API ready for trading")
    yield
    
    # Shutdown
    logger.info("🛑 UCB API stopping...")
    app.state.persistence_task.cancel()
    
    # Save final state
    async with app.state.lock:
        app.state.ucb.save_state()
    logger.info("💾 Final state saved")

app = FastAPI(title="UCB Service", lifespan=lifespan)

# ==============================================
# CORS CONFIGURATION (EXPLICIT)
# ==============================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # No CORS in production
    allow_methods=[],
    allow_headers=[]
)

# ==============================================
# HELPER FUNCTIONS
# ==============================================
def _nz(x):
    """Safely convert to float, handling None and NaN"""
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)

async def state_persistence_loop(ucb, lock, interval=60):
    """Auto-save state every interval seconds with lock"""
    while True:
        await asyncio.sleep(interval)
        async with lock:
            ucb.save_state()
            logger.info(f"Auto-saved state: PnL=${ucb.model.daily_pnl:.2f}")

# ==============================================
# API ENDPOINTS WITH PROPER LOCKING
# ==============================================

# Maximum request body size (64KB)
MAX_BODY_BYTES = 64 * 1024

@app.post("/ucb/recommend")
async def recommend(
    req: Request, 
    content_type: str = Header(None),
    x_req_id: str = Header(None, alias="X-Req-Id")
):
    """Get trading recommendation based on market data"""
    # Validate content type
    if not (content_type and "application/json" in content_type.lower()):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    
    body = await req.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large")
    
    if x_req_id:
        logger.info(f"[{x_req_id}] Processing recommendation request ({len(body)} bytes)")
    
    # Read-only operation, but optionally lock for absolute serialization
    # async with app.state.lock:  # Uncomment if you want strict serialization
    rec_json = app.state.ucb.get_recommendation(body.decode("utf-8"))
    
    result = json.loads(rec_json)
    
    if x_req_id:
        logger.info(f"[{x_req_id}] Recommendation: {result.get('strategy')} | Size: {result.get('position_size')}")
    
    return JSONResponse(content=result)

@app.post("/ucb/update_pnl")
async def update_pnl(
    req: Request,
    x_req_id: str = Header(None, alias="X-Req-Id")
):
    """Update model with trade PnL - LOCKED FOR SAFETY"""
    data = await req.json()
    strategy = data.get("strategy", "")
    pnl = float(data.get("pnl", 0.0))
    
    if x_req_id:
        logger.info(f"[{x_req_id}] PnL update: {strategy} = ${pnl:.2f}")
    
    # Lock for state mutation
    async with app.state.lock:
        out = app.state.ucb.update_pnl(strategy, pnl)
        app.state.ucb.save_state()  # Use UCBIntegration's save method
    
    return JSONResponse(content=json.loads(out))

@app.post("/ucb/reset_daily")
async def reset_daily(x_req_id: str = Header(None, alias="X-Req-Id")):
    """Reset daily statistics - LOCKED FOR SAFETY"""
    if x_req_id:
        logger.info(f"[{x_req_id}] Daily reset requested")
    
    # Lock for state mutation
    async with app.state.lock:
        out = app.state.ucb.reset_daily()
        app.state.ucb.save_state()  # Use UCBIntegration's save method
    
    return JSONResponse(content=json.loads(out))

@app.get("/ucb/limits")
async def limits():
    """Get current risk limits without market data - READ ONLY"""
    trade_blocked, msg, level = app.state.ucb.model.should_stop_trading()
    
    return JSONResponse(content={
        "can_trade": not trade_blocked,
        "reason": msg if trade_blocked else "OK",
        "warning": msg if level == "warning" else None,
        "current_drawdown": _nz(app.state.ucb.model.current_drawdown),
        "daily_pnl": _nz(app.state.ucb.model.daily_pnl),
        "account_balance": _nz(app.state.ucb.model.account_balance)
    })

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "UCB Trading API",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": hasattr(app.state, 'ucb')
    }

@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint - READ ONLY"""
    if not hasattr(app.state, 'ucb'):
        return {"error": "Model not loaded"}
    
    stats = {}
    for sid, perf in app.state.ucb.model.strategy_stats.items():
        # Find strategy name
        for strat in ["opening_drive", "vwap_reversion", "correlation_divergence", 
                     "closing_squeeze", "momentum_continuation"]:
            if app.state.ucb.model._stable_sid(strat) == sid:
                stats[strat] = {
                    "trades": perf.get("trades", 0),
                    "win_rate": perf["wins"] / max(perf["trades"], 1) if "wins" in perf else 0,
                    "total_pnl": perf.get("total_pnl", 0)
                }
                break
    
    return {
        "strategies": stats,
        "daily_pnl": _nz(app.state.ucb.model.daily_pnl),
        "current_drawdown": _nz(app.state.ucb.model.current_drawdown),
        "account_balance": _nz(app.state.ucb.model.account_balance),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    # BIND TO LOCALHOST ONLY
    uvicorn.run(
        app, 
        host="127.0.0.1",  # Localhost only
        port=5000, 
        timeout_keep_alive=5,
        log_level="info",
        access_log=True
    )
