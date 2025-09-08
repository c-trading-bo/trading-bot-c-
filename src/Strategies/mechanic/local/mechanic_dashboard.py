#!/usr/bin/env python3
"""
MECHANIC DASHBOARD - Integrated with your existing dashboard
STRICTLY LOCAL ONLY - No cloud features
"""

from flask import Flask, jsonify, render_template_string
import json
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import sys
import os

# Add the parent directory to path to import bot_mechanic
sys.path.append(str(Path(__file__).parent))
from bot_mechanic import LocalBotMechanic

class MechanicDashboard:
    def __init__(self, port=5051):
        self.port = port
        self.mechanic = LocalBotMechanic()
        self.app = Flask(__name__)
        self.setup_routes()
        self.start_background_monitoring()
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/mechanic/health')
        def health():
            """Health endpoint for mechanic"""
            try:
                data = self.mechanic.get_dashboard_data()
                return jsonify({
                    'status': 'healthy',
                    'mechanic': data,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
        
        @self.app.route('/mechanic/scan')
        def trigger_scan():
            """Trigger full scan"""
            try:
                results = self.mechanic.deep_scan(verbose=False)
                return jsonify({
                    'status': 'success',
                    'scan_results': results,
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
        
        @self.app.route('/mechanic/fix')
        def auto_fix():
            """Trigger auto-fix"""
            try:
                self.mechanic.auto_fix_all()
                return jsonify({
                    'status': 'success',
                    'message': 'Auto-fix completed',
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
        
        @self.app.route('/mechanic/dashboard')
        def dashboard():
            """Mechanic dashboard page"""
            return render_template_string(self.get_dashboard_html())
        
        @self.app.route('/mechanic/api/status')
        def api_status():
            """API endpoint for status updates"""
            try:
                quick_health = self.mechanic.quick_scan()
                dashboard_data = self.mechanic.get_dashboard_data()
                
                return jsonify({
                    'mechanic_status': 'active',
                    'health_score': quick_health['health_score'],
                    'issues_count': quick_health['issues'],
                    'is_healthy': quick_health['healthy'],
                    'last_scan': dashboard_data.get('last_scan', {}),
                    'recent_repairs': dashboard_data.get('recent_repairs', []),
                    'feature_count': dashboard_data.get('feature_count', 0),
                    'categories': quick_health.get('checks', {}),
                    'monitoring_active': True,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({
                    'mechanic_status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }), 500
    
    def start_background_monitoring(self):
        """Start background monitoring thread"""
        def monitor_loop():
            while True:
                try:
                    # Quick health check every 5 minutes
                    self.mechanic.quick_scan()
                    time.sleep(300)
                except Exception as e:
                    print(f"Monitor error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def get_dashboard_html(self):
        """Get HTML for mechanic dashboard"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Local Bot Mechanic Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .header { background: #333; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .card { background: #2a2a2a; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .status-healthy { color: #4CAF50; }
        .status-warning { color: #FF9800; }
        .status-error { color: #F44336; }
        .button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .button:hover { background: #005a8b; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .metric { background: #333; padding: 15px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; }
        .logs { background: #1a1a1a; padding: 15px; border-radius: 8px; font-family: monospace; max-height: 300px; overflow-y: auto; }
        #status-indicator { width: 20px; height: 20px; border-radius: 50%; display: inline-block; margin-right: 10px; }
    </style>
    <script>
        function updateStatus() {
            fetch('/mechanic/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update status indicator
                    const indicator = document.getElementById('status-indicator');
                    const statusText = document.getElementById('status-text');
                    
                    if (data.is_healthy) {
                        indicator.style.backgroundColor = '#4CAF50';
                        statusText.textContent = 'Healthy';
                        statusText.className = 'status-healthy';
                    } else {
                        indicator.style.backgroundColor = '#FF9800';
                        statusText.textContent = `${data.issues_count} Issues`;
                        statusText.className = 'status-warning';
                    }
                    
                    // Update metrics
                    document.getElementById('health-score').textContent = data.health_score + '%';
                    document.getElementById('feature-count').textContent = data.feature_count;
                    document.getElementById('issues-count').textContent = data.issues_count;
                    document.getElementById('last-update').textContent = new Date(data.timestamp).toLocaleTimeString();
                    
                    // Update category status
                    const categoryStatus = document.getElementById('category-status');
                    categoryStatus.innerHTML = '';
                    
                    for (const [category, status] of Object.entries(data.categories || {})) {
                        const statusClass = status.healthy ? 'status-healthy' : 'status-warning';
                        categoryStatus.innerHTML += `<div><span class="${statusClass}">●</span> ${category}: ${status.healthy ? 'OK' : status.issue}</div>`;
                    }
                    
                    // Update recent repairs
                    const repairsList = document.getElementById('recent-repairs');
                    repairsList.innerHTML = '';
                    
                    (data.recent_repairs || []).forEach(repair => {
                        const statusIcon = repair.success ? '✅' : '❌';
                        const time = new Date(repair.timestamp).toLocaleString();
                        repairsList.innerHTML += `<div>${statusIcon} ${repair.system} (${time})</div>`;
                    });
                })
                .catch(error => {
                    console.error('Status update failed:', error);
                    document.getElementById('status-indicator').style.backgroundColor = '#F44336';
                    document.getElementById('status-text').textContent = 'Error';
                    document.getElementById('status-text').className = 'status-error';
                });
        }
        
        function triggerScan() {
            document.getElementById('scan-btn').textContent = 'Scanning...';
            fetch('/mechanic/scan')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('scan-btn').textContent = 'Full Scan';
                    updateStatus();
                    alert('Scan completed!');
                })
                .catch(error => {
                    document.getElementById('scan-btn').textContent = 'Full Scan';
                    alert('Scan failed: ' + error);
                });
        }
        
        function triggerFix() {
            document.getElementById('fix-btn').textContent = 'Fixing...';
            fetch('/mechanic/fix')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fix-btn').textContent = 'Auto-Fix';
                    updateStatus();
                    alert('Auto-fix completed!');
                })
                .catch(error => {
                    document.getElementById('fix-btn').textContent = 'Auto-Fix';
                    alert('Fix failed: ' + error);
                });
        }
        
        // Update status every 30 seconds
        setInterval(updateStatus, 30000);
        
        // Initial update
        window.onload = updateStatus;
    </script>
</head>
<body>
    <div class="header">
        <h1>🧠 Local Bot Mechanic Dashboard</h1>
        <div style="display: flex; align-items: center;">
            <span id="status-indicator"></span>
            <span id="status-text">Loading...</span>
            <span style="margin-left: auto;">Last Update: <span id="last-update">--:--:--</span></span>
        </div>
    </div>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-value" id="health-score">--</div>
            <div>Health Score</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="feature-count">--</div>
            <div>Features Tracked</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="issues-count">--</div>
            <div>Active Issues</div>
        </div>
    </div>
    
    <div class="card">
        <h3>Controls</h3>
        <button class="button" id="scan-btn" onclick="triggerScan()">Full Scan</button>
        <button class="button" id="fix-btn" onclick="triggerFix()">Auto-Fix</button>
        <button class="button" onclick="updateStatus()">Refresh Status</button>
    </div>
    
    <div class="card">
        <h3>System Status</h3>
        <div id="category-status">Loading...</div>
    </div>
    
    <div class="card">
        <h3>Recent Repairs</h3>
        <div id="recent-repairs">Loading...</div>
    </div>
    
    <div class="card">
        <h3>Integration Status</h3>
        <div>✅ Dashboard Integration: Active</div>
        <div>✅ Background Monitoring: Running</div>
        <div>✅ Auto-Repair: Enabled</div>
        <div>✅ Health Checks: Every 5 minutes</div>
    </div>
</body>
</html>
        """
    
    def run(self, debug=False):
        """Run the dashboard"""
        print(f"\n🚀 Starting Mechanic Dashboard on port {self.port}")
        print(f"📊 Dashboard URL: http://localhost:{self.port}/mechanic/dashboard")
        print(f"🔗 API Health: http://localhost:{self.port}/mechanic/health")
        
        try:
            self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)
        except Exception as e:
            print(f"❌ Dashboard failed to start: {e}")
            print(f"💡 Try changing port or check if port {self.port} is already in use")

def main():
    """Main entry point for dashboard"""
    dashboard = MechanicDashboard(port=5051)
    dashboard.run()

if __name__ == "__main__":
    main()
