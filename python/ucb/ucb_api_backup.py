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
    finally:
        # Cleanup on shutdown
        if ucb and hasattr(ucb.model, '_save_state'):
            try:
                ucb.model._save_state()
                print("💾 [FastAPI] State saved on shutdown")
            except Exception as e:
                print(f"⚠️ [FastAPI] Error saving state: {e}")

app = FastAPI(
    title="UCB Trading Service",
    description="Neural UCB for TopStep ES/NQ Trading",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "UCB Trading Service", "ready": ucb is not None}

@app.get("/health")
async def health():
    """Health check endpoint"""
    if ucb is None:
        raise HTTPException(status_code=503, detail="UCB not initialized")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "daily_pnl": ucb.model.daily_pnl,
        "current_drawdown": ucb.model.current_drawdown,
        "strategy_count": len(ucb.model.strategy_stats)
    }

@app.post("/ucb/recommend")
async def recommend(req: Request):
    """Get trading recommendation from UCB model"""
    if ucb is None:
        raise HTTPException(status_code=503, detail="UCB not initialized")
    
    try:
        body = await req.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty request body")
        
        # Forward raw JSON string directly to Python model
        rec_json = ucb.get_recommendation(body.decode("utf-8"))
        rec_dict = json.loads(rec_json)
        
        return JSONResponse(content=rec_dict)
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        print(f"❌ [FastAPI] Error in recommend: {e}")
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"trade": False, "reason": f"Internal error: {str(e)}"}
        )

@app.post("/ucb/update_pnl")
async def update_pnl(req: Request):
    """Update strategy P&L for learning"""
    if ucb is None:
        raise HTTPException(status_code=503, detail="UCB not initialized")
    
    try:
        data = await req.json()
        strategy = data.get("strategy", "")
        pnl = float(data.get("pnl", 0.0))
        
        if not strategy:
            raise HTTPException(status_code=400, detail="Missing strategy parameter")
        
        out = ucb.update_pnl(strategy, pnl)
        result = json.loads(out)
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid pnl value: {str(e)}")
    except Exception as e:
        print(f"❌ [FastAPI] Error in update_pnl: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/ucb/reset_daily")
async def reset_daily():
    """Reset daily P&L and drawdown stats"""
    if ucb is None:
        raise HTTPException(status_code=503, detail="UCB not initialized")
    
    try:
        out = ucb.reset_daily()
        result = json.loads(out)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"❌ [FastAPI] Error in reset_daily: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/ucb/stats")
async def get_stats():
    """Get current UCB statistics"""
    if ucb is None:
        raise HTTPException(status_code=503, detail="UCB not initialized")
    
    try:
        stats = {
            "daily_pnl": ucb.model.daily_pnl,
            "current_drawdown": ucb.model.current_drawdown,
            "account_balance": ucb.model.account_balance,
            "strategy_stats": ucb.model.strategy_stats,
            "active_strategies": len(ucb.model.strategy_stats),
            "compliance_ok": ucb.model.should_stop_trading()[0] != True
        }
        
        return JSONResponse(content=stats)
        
    except Exception as e:
        print(f"❌ [FastAPI] Error in get_stats: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    # Production settings
    import logging
    logging.basicConfig(level=logging.INFO)
    
    host = os.getenv("UCB_HOST", "0.0.0.0")
    port = int(os.getenv("UCB_PORT", "5000"))
    
    print(f"🚀 Starting UCB FastAPI server on {host}:{port}")
    print("⚠️  IMPORTANT: Keep this single-process (no --workers > 1)")
    print("📊 UCB stats live in memory and sync across requests")
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        timeout_keep_alive=5,  # Fast timeout for production
        access_log=True,
        log_level="info"
    )
