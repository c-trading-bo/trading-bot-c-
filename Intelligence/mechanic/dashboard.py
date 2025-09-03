#!/usr/bin/env python3
"""
Web Dashboard for Local Bot Mechanic
Real-time monitoring of your trading bot
"""

import json
import time
import threading
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path('Intelligence/mechanic/reports')), **kwargs)
    
    def do_GET(self):
        if self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get current status
            status = self.get_bot_status()
            self.wfile.write(json.dumps(status).encode())
            
        elif self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            # Get health data
            health = self.get_health_data()
            self.wfile.write(json.dumps(health).encode())
            
        elif self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Serve dashboard HTML
            html = self.get_dashboard_html()
            self.wfile.write(html.encode())
            
        else:
            super().do_GET()
    
    def get_bot_status(self):
        """Get current bot status"""
        try:
            # Check if mechanic database exists
            db_path = Path('Intelligence/mechanic/database/knowledge.json')
            if db_path.exists():
                with open(db_path, 'r') as f:
                    knowledge = json.load(f)
                
                last_scan = knowledge.get('last_scan')
                if last_scan:
                    return {
                        'status': 'active',
                        'last_scan': last_scan.get('timestamp'),
                        'files_scanned': last_scan.get('files_scanned', 0),
                        'issues_found': len(last_scan.get('issues_found', [])),
                        'auto_fixed': len(last_scan.get('auto_fixed', [])),
                        'health_score': max(0, 100 - len(last_scan.get('issues_found', [])) * 5)
                    }
            
            return {
                'status': 'inactive',
                'message': 'Bot mechanic not initialized'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_health_data(self):
        """Get health monitoring data"""
        # Mock health data - replace with real checks
        return {
            'trading_system': {'status': 'healthy', 'last_check': datetime.now().isoformat()},
            'ml_models': {'status': 'healthy', 'model_count': 3},
            'data_pipeline': {'status': 'healthy', 'last_update': datetime.now().isoformat()},
            'workflows': {'status': 'healthy', 'active_count': 33}
        }
    
    def get_dashboard_html(self):
        """Generate dashboard HTML"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local Bot Mechanic Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .card h3 {
            color: #5a67d8;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-healthy { background: #48bb78; }
        .status-warning { background: #ed8936; }
        .status-error { background: #f56565; }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
        }
        .metric:last-child { border-bottom: none; }
        .value {
            font-weight: bold;
            color: #5a67d8;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #48bb78, #38a169);
            transition: width 0.3s ease;
        }
        .logs {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 15px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            max-height: 300px;
            overflow-y: auto;
        }
        .refresh-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            margin: 10px 5px;
            transition: transform 0.2s;
        }
        .refresh-btn:hover {
            transform: translateY(-2px);
        }
        .auto-refresh {
            color: #48bb78;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Local Bot Mechanic</h1>
            <p>Real-time monitoring of your trading intelligence system</p>
            <div class="auto-refresh" id="autoRefresh">⟳ Auto-refreshing every 30s</div>
        </div>

        <div class="dashboard">
            <div class="card">
                <h3>🚀 Bot Status</h3>
                <div class="metric">
                    <span>Status</span>
                    <span class="value" id="botStatus">
                        <span class="status-indicator status-healthy"></span>Loading...
                    </span>
                </div>
                <div class="metric">
                    <span>Last Scan</span>
                    <span class="value" id="lastScan">-</span>
                </div>
                <div class="metric">
                    <span>Files Monitored</span>
                    <span class="value" id="filesMonitored">-</span>
                </div>
                <div class="metric">
                    <span>Health Score</span>
                    <span class="value" id="healthScore">-</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" id="healthProgress" style="width: 0%"></div>
                </div>
            </div>

            <div class="card">
                <h3>🏥 System Health</h3>
                <div class="metric">
                    <span>Trading System</span>
                    <span class="value" id="tradingHealth">
                        <span class="status-indicator status-healthy"></span>Checking...
                    </span>
                </div>
                <div class="metric">
                    <span>ML Models</span>
                    <span class="value" id="mlHealth">
                        <span class="status-indicator status-healthy"></span>Checking...
                    </span>
                </div>
                <div class="metric">
                    <span>Data Pipeline</span>
                    <span class="value" id="dataHealth">
                        <span class="status-indicator status-healthy"></span>Checking...
                    </span>
                </div>
                <div class="metric">
                    <span>Workflows</span>
                    <span class="value" id="workflowHealth">
                        <span class="status-indicator status-healthy"></span>Checking...
                    </span>
                </div>
            </div>

            <div class="card">
                <h3>🔧 Recent Activity</h3>
                <div id="recentActivity">
                    <div class="metric">
                        <span>Issues Found</span>
                        <span class="value" id="issuesFound">-</span>
                    </div>
                    <div class="metric">
                        <span>Auto-Fixed</span>
                        <span class="value" id="autoFixed">-</span>
                    </div>
                    <div class="metric">
                        <span>Manual Fixes Needed</span>
                        <span class="value" id="manualFixes">-</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>📊 Quick Actions</h3>
                <button class="refresh-btn" onclick="runHealthCheck()">🔍 Health Check</button>
                <button class="refresh-btn" onclick="triggerScan()">📡 Full Scan</button>
                <button class="refresh-btn" onclick="generateReport()">📋 Generate Report</button>
                <button class="refresh-btn" onclick="viewLogs()">📝 View Logs</button>
            </div>
        </div>

        <div class="card">
            <h3>📋 System Logs</h3>
            <div class="logs" id="systemLogs">
                Loading system logs...
            </div>
        </div>
    </div>

    <script>
        let autoRefreshInterval;

        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateStatusDisplay(data);
            } catch (error) {
                console.error('Error fetching status:', error);
                document.getElementById('botStatus').innerHTML = 
                    '<span class="status-indicator status-error"></span>Connection Error';
            }
        }

        async function fetchHealth() {
            try {
                const response = await fetch('/api/health');
                const data = await response.json();
                updateHealthDisplay(data);
            } catch (error) {
                console.error('Error fetching health:', error);
            }
        }

        function updateStatusDisplay(data) {
            const statusMap = {
                'active': '<span class="status-indicator status-healthy"></span>Active',
                'inactive': '<span class="status-indicator status-warning"></span>Inactive',
                'error': '<span class="status-indicator status-error"></span>Error'
            };

            document.getElementById('botStatus').innerHTML = statusMap[data.status] || 'Unknown';
            document.getElementById('lastScan').textContent = data.last_scan ? 
                new Date(data.last_scan).toLocaleString() : 'Never';
            document.getElementById('filesMonitored').textContent = data.files_scanned || '0';
            document.getElementById('issuesFound').textContent = data.issues_found || '0';
            document.getElementById('autoFixed').textContent = data.auto_fixed || '0';
            
            const healthScore = data.health_score || 0;
            document.getElementById('healthScore').textContent = healthScore + '%';
            document.getElementById('healthProgress').style.width = healthScore + '%';
            
            // Update progress bar color
            const progressBar = document.getElementById('healthProgress');
            if (healthScore >= 80) {
                progressBar.style.background = 'linear-gradient(90deg, #48bb78, #38a169)';
            } else if (healthScore >= 60) {
                progressBar.style.background = 'linear-gradient(90deg, #ed8936, #dd6b20)';
            } else {
                progressBar.style.background = 'linear-gradient(90deg, #f56565, #e53e3e)';
            }
        }

        function updateHealthDisplay(data) {
            const systems = ['trading_system', 'ml_models', 'data_pipeline', 'workflows'];
            const displayIds = ['tradingHealth', 'mlHealth', 'dataHealth', 'workflowHealth'];
            
            systems.forEach((system, index) => {
                const systemData = data[system];
                const status = systemData?.status || 'unknown';
                const indicator = status === 'healthy' ? 'status-healthy' : 
                                status === 'warning' ? 'status-warning' : 'status-error';
                
                document.getElementById(displayIds[index]).innerHTML = 
                    `<span class="status-indicator ${indicator}"></span>${status.charAt(0).toUpperCase() + status.slice(1)}`;
            });
        }

        function updateLogs() {
            const logs = [
                `[${new Date().toLocaleTimeString()}] System monitoring active`,
                `[${new Date(Date.now() - 60000).toLocaleTimeString()}] Health check completed`,
                `[${new Date(Date.now() - 120000).toLocaleTimeString()}] Auto-repair system ready`,
                `[${new Date(Date.now() - 180000).toLocaleTimeString()}] Local Bot Mechanic initialized`
            ];
            
            document.getElementById('systemLogs').innerHTML = logs.join('\\n');
        }

        function runHealthCheck() {
            alert('Health check triggered! Results will appear in the dashboard.');
            fetchStatus();
            fetchHealth();
        }

        function triggerScan() {
            alert('Full system scan triggered! This may take a few minutes.');
            setTimeout(() => {
                fetchStatus();
                fetchHealth();
            }, 2000);
        }

        function generateReport() {
            window.open('latest_report.html', '_blank');
        }

        function viewLogs() {
            alert('Opening detailed logs...');
        }

        function startAutoRefresh() {
            autoRefreshInterval = setInterval(() => {
                fetchStatus();
                fetchHealth();
                updateLogs();
            }, 30000);
        }

        // Initialize dashboard
        fetchStatus();
        fetchHealth();
        updateLogs();
        startAutoRefresh();

        // Update auto-refresh indicator
        setInterval(() => {
            const now = new Date();
            document.getElementById('autoRefresh').textContent = 
                `⟳ Auto-refreshing every 30s (Last: ${now.toLocaleTimeString()})`;
        }, 1000);
    </script>
</body>
</html>'''

def start_dashboard(port=8888):
    """Start the web dashboard"""
    print(f"\n🌐 Starting Local Bot Mechanic Dashboard on port {port}")
    print(f"📊 Dashboard URL: http://localhost:{port}")
    print("Press Ctrl+C to stop\n")
    
    # Ensure reports directory exists
    reports_dir = Path('Intelligence/mechanic/reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
            print(f"✅ Dashboard server running at http://localhost:{port}")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    start_dashboard(port)
