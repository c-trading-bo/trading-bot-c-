#!/usr/bin/env python3
"""
Simple dashboard test server to test trading bot dashboard functionality
without requiring the full bot infrastructure.
"""

import json
import datetime
import random
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import sys
import os

class DashboardTestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path == '/':
            # Redirect to dashboard
            self.send_response(302)
            self.send_header('Location', '/dashboard.html')
            self.end_headers()
            return
        
        if path == '/dashboard':
            # Serve dashboard.html
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('wwwroot/dashboard.html', 'rb') as f:
                self.wfile.write(f.read())
            return
        
        if path == '/health/system':
            # Mock health endpoint
            health_data = {
                "status": "HEALTHY",
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "checks": {
                    "mlPersistence": {
                        "status": "Healthy",
                        "message": "All models saved successfully",
                        "checkTime": datetime.datetime.utcnow().isoformat() + "Z"
                    },
                    "strategyConfig": {
                        "status": "Healthy", 
                        "message": "All strategies configured correctly",
                        "checkTime": datetime.datetime.utcnow().isoformat() + "Z"
                    },
                    "sessionWindows": {
                        "status": "Healthy",
                        "message": "Session timers functioning properly", 
                        "checkTime": datetime.datetime.utcnow().isoformat() + "Z"
                    },
                    "dataFeeds": {
                        "status": "Healthy",
                        "message": "Real-time data flowing",
                        "checkTime": datetime.datetime.utcnow().isoformat() + "Z"
                    },
                    "riskManagement": {
                        "status": "Warning" if random.random() < 0.3 else "Healthy",
                        "message": "Daily loss approaching 70% limit" if random.random() < 0.3 else "Risk controls operating normally",
                        "checkTime": datetime.datetime.utcnow().isoformat() + "Z"
                    },
                    "orderRouting": {
                        "status": "Healthy",
                        "message": "Connection to TopstepX stable (PAPER)",
                        "checkTime": datetime.datetime.utcnow().isoformat() + "Z"
                    },
                    "strategySignals": {
                        "status": "Healthy",
                        "message": "All signals processing correctly",
                        "checkTime": datetime.datetime.utcnow().isoformat() + "Z"
                    },
                    "positionTracking": {
                        "status": "Healthy",
                        "message": "Position sync verified",
                        "checkTime": datetime.datetime.utcnow().isoformat() + "Z"
                    },
                    "priceValidation": {
                        "status": "Healthy",
                        "message": "Price validation passing", 
                        "checkTime": datetime.datetime.utcnow().isoformat() + "Z"
                    }
                },
                "details": {
                    "accountId": 12345,
                    "mode": "PAPER",
                    "uptime": 1234.5,
                    "userHub": "Connected",
                    "marketHub": "Connected",
                    "learnerStatus": "Active"
                }
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(health_data).encode())
            return
            
        if path == '/stream/metrics':
            # Server-sent events for real-time data
            self.send_response(200)
            self.send_header('Content-type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Send hello event
            hello_data = {"ok": True}
            self.wfile.write(f"event: hello\ndata: {json.dumps(hello_data)}\n\n".encode())
            self.wfile.flush()
            
            # Keep sending mock data every 2 seconds
            try:
                while True:
                    mock_data = self.generate_mock_metrics()
                    data_str = json.dumps(mock_data)
                    self.wfile.write(f"data: {data_str}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(2)
            except (ConnectionResetError, BrokenPipeError):
                # Client disconnected
                pass
            return
        
        if path == '/api/metrics':
            # Polling endpoint for dashboard metrics
            mock_data = self.generate_mock_metrics()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(mock_data).encode())
            return
        
        if path in ['/api/bot/start', '/api/bot/stop', '/api/bot/mode']:
            # Mock bot control endpoints
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"success": True, "message": f"Mock {path} executed successfully"}
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Default handling for static files
        super().do_GET()
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        if path in ['/api/bot/start', '/api/bot/stop', '/api/bot/mode']:
            # Mock bot control endpoints
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"success": True, "message": f"Mock {path} executed successfully"}
            self.wfile.write(json.dumps(response).encode())
            return
        
        super().do_POST()
    
    def generate_mock_metrics(self):
        now = datetime.datetime.utcnow()
        
        # Generate realistic looking data
        base_pnl = 125.50 + random.uniform(-50, 50)
        unrealized = random.uniform(-100, 100)
        
        return {
            "accountId": 12345,
            "mode": "PAPER",
            "realized": round(base_pnl, 2),
            "unrealized": round(unrealized, 2),
            "day": round(base_pnl + unrealized, 2),
            "maxDailyLoss": -2000,
            "remaining": round(1920.25 + base_pnl + unrealized, 2),
            "userHub": "Connected",
            "marketHub": "Connected",
            "localTime": now.isoformat() + "Z",
            "positions": [
                {
                    "sym": "ES",
                    "qty": random.choice([0, 1, 2, -1, -2]),
                    "avg": round(5000 + random.uniform(-10, 10), 2),
                    "mark": round(5000 + random.uniform(-5, 15), 2),
                    "uPnL": round(random.uniform(-100, 100), 2),
                    "rPnL": 0
                }
            ],
            "curfewNoNew": False,
            "dayPnlNoNew": False,
            "allowedNow": ["S2", "S3", "S6", "S11"],
            "learnerOn": True,
            "learnerLastRun": (now - datetime.timedelta(minutes=random.randint(5, 30))).isoformat() + "Z",
            "learnerApplied": True,
            "learnerNote": "Model updated successfully",
            "strategyPnl": {
                "S2": {"trades": 5, "pnl": round(random.uniform(-50, 150), 2), "winRate": round(random.uniform(60, 90), 1)},
                "S3": {"trades": 3, "pnl": round(random.uniform(-100, 100), 2), "winRate": round(random.uniform(30, 70), 1)},
                "S6": {"trades": 2, "pnl": round(random.uniform(-25, 125), 2), "winRate": round(random.uniform(50, 100), 1)},
                "S11": {"trades": 4, "pnl": round(random.uniform(-75, 75), 2), "winRate": round(random.uniform(40, 80), 1)}
            },
            "healthStatus": "HEALTHY",
            "healthDetails": {
                "mlPersistence": {"status": "Healthy", "message": "All models saved successfully"},
                "strategyConfig": {"status": "Healthy", "message": "All strategies configured correctly"},
                "dataFeeds": {"status": "Healthy", "message": "Real-time data flowing"}
            },
            "overview": {
                "accountBalance": 100000 + round(base_pnl + unrealized, 2),
                "totalPnL": round(base_pnl + unrealized, 2),
                "openPositions": random.randint(0, 3),
                "todayTrades": random.randint(10, 25),
                "botMode": "PAPER",
                "activeStrategy": "Multiple"
            },
            "learning": {
                "status": "Active",
                "currentLoop": now.strftime("%H:%M:%S"),
                "learningRate": "0.001",
                "lastAdaptation": (now - datetime.timedelta(minutes=2)).strftime("%H:%M:%S"),
                "accuracy": round(85.3 + random.uniform(-5, 5), 1),
                "adaptationScore": "High",
                "modelConfidence": round(92.1 + random.uniform(-3, 3), 1),
                "cycles": 247 + random.randint(0, 10),
                "progress": random.randint(50, 95),
                "stages": [
                    {"name": "Data Analysis", "progress": 100, "active": True},
                    {"name": "Pattern Recognition", "progress": random.randint(85, 100), "active": True},
                    {"name": "Model Training", "progress": random.randint(60, 90), "active": random.choice([True, False])},
                    {"name": "Validation", "progress": random.randint(0, 70), "active": random.choice([True, False])}
                ]
            },
            "system": {
                "uptime": f"{random.randint(0, 2)}d {random.randint(0, 23)}h {random.randint(0, 59)}m",
                "dataQuality": f"{round(98.5 + random.uniform(-1.5, 1.5), 1)}%",
                "cpuUsage": random.randint(20, 60),
                "memoryUsage": random.randint(40, 80)
            },
            "timestamp": now.isoformat() + "Z"
        }

def main():
    port = 5050
    print(f"🎯 Starting Dashboard Test Server on port {port}")
    print(f"📊 Visit http://localhost:{port}/dashboard to test the dashboard")
    print(f"🔄 Press Ctrl+C to stop")
    
    # Change to the repository root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    server = HTTPServer(('localhost', port), DashboardTestHandler)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard test server...")
        server.shutdown()

if __name__ == '__main__':
    main()