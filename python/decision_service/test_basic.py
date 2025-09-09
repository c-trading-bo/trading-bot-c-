#!/usr/bin/env python3
"""
Simplified Decision Service test using basic HTTP client
Tests the core decision logic without external dependencies
"""

import sys
import json
import time
from datetime import datetime, timezone
from urllib.request import urlopen, Request, HTTPError
from urllib.parse import urljoin

BASE_URL = "http://127.0.0.1:7080"

def test_health():
    """Test health endpoint"""
    print("🩺 Testing health endpoint...")
    try:
        req = Request(f"{BASE_URL}/health")
        with urlopen(req, timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"✅ Health: {data.get('status')} - Regime: {data.get('regime')} - Daily P&L: ${data.get('daily_pnl', 0):.2f}")
                return True
            else:
                print(f"❌ Health check failed: {response.status}")
                return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_signal():
    """Test /v1/signal endpoint - the main decision logic"""
    print("\n🎯 Testing signal decision endpoint...")
    try:
        signal_data = {
            "symbol": "ES",
            "strategyId": "S2",
            "side": "LONG",
            "signalId": f"es-s2-{int(time.time())}-001",
            "hints": {"stopPoints": 3.0},
            "cloud": {
                "p": 0.67,
                "sourceModelId": "cloud.es.s2.v23",
                "latencyMs": 18
            }
        }
        
        json_data = json.dumps(signal_data).encode('utf-8')
        req = Request(f"{BASE_URL}/v1/signal", data=json_data)
        req.add_header('Content-Type', 'application/json')
        
        with urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"✅ Signal decision:")
                print(f"   Gate: {data.get('gate')}")
                print(f"   Regime: {data.get('regime')}")
                print(f"   P_cloud: {data.get('p_cloud', 0):.3f}")
                print(f"   P_online: {data.get('p_online', 0):.3f}")
                print(f"   P_final: {data.get('p_final', 0):.3f}")
                print(f"   UCB: {data.get('ucb', 0):.3f}")
                print(f"   Proposed: {data.get('proposedContracts', 0)} → Final: {data.get('finalContracts', 0)}")
                print(f"   Decision ID: {data.get('decisionId')}")
                
                if data.get('gate'):
                    return data.get('decisionId')
                else:
                    print(f"   Reason: {data.get('reason')}")
                    return None
            else:
                print(f"❌ Signal processing failed: {response.status}")
                return None
    except Exception as e:
        print(f"❌ Signal processing error: {e}")
        return None

def main():
    print("🧠 ML/RL Decision Service Basic Test")
    print("Testing core endpoints without external dependencies")
    print("="*50)
    
    # Test health
    if not test_health():
        print("❌ Health check failed - is the Decision Service running?")
        print("\nTo start the service:")
        print("cd python/decision_service")
        print("python decision_service.py")
        return False
    
    # Test signal
    decision_id = test_signal()
    if decision_id:
        print(f"\n🎉 Decision Service is working! Decision ID: {decision_id}")
    else:
        print("\n⚠️ Decision Service responded but gate=false or error")
    
    print("\n" + "="*50)
    print("✅ Basic test complete!")
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)