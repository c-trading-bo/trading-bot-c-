#!/usr/bin/env python3
"""
LOCAL BOT MECHANIC - Auto-Start with Dashboard Integration
Automatically launches with trading bot and integrates with main dashboard
"""

import sys
import os
import json
import time
import threading
import subprocess
import signal
from datetime import datetime
from pathlib import Path

# Centralized URL configuration to eliminate hardcoded URLs
class LocalMechanicConfig:
    def __init__(self, host="localhost", dashboard_port=5051):
        self.host = host
        self.dashboard_port = dashboard_port
    
    def get_dashboard_url(self):
        return f"http://{self.host}:{self.dashboard_port}"
    
    def get_iframe_src(self):
        return f"http://{self.host}:{self.dashboard_port}"

class MechanicAutoLauncher:
    def __init__(self):
        self.version = "3.0.0-AUTO"
        self.running = True
        self.mechanic_process = None
        self.flask_app = None
        self.flask_thread = None
        self.monitor_thread = None
        self.base_path = Path.cwd()
        self.mechanic_path = self.base_path / "Intelligence" / "mechanic" / "local"
        
        # Dashboard integration settings with centralized config
        self.config = LocalMechanicConfig()
        self.dashboard_port = 5051  # Different from main dashboard (5050)
        self.check_interval = 30    # Health check every 30 seconds
        
        print(f"🚀 Local Bot Mechanic Auto-Launcher v{self.version}")
        print(f"📍 Base path: {self.base_path}")
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.stop()
    
    def check_dependencies(self):
        """Check and install required dependencies"""
        required = {
            'pandas': 'pandas',
            'numpy': 'numpy', 
            'sklearn': 'scikit-learn',
            'requests': 'requests',
            'yfinance': 'yfinance',
            'flask': 'flask'
        }
        
        print("📦 Checking dependencies...")
        missing = []
        for import_name, pip_name in required.items():
            try:
                __import__(import_name)
                print(f"  ✅ {import_name}")
            except ImportError:
                missing.append(pip_name)
                print(f"  ❌ {import_name} missing")
        
        if missing:
            print(f"\n📦 Installing {len(missing)} missing packages...")
            for package in missing:
                try:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 check=True, capture_output=True)
                    print(f"  ✅ Installed {package}")
                except subprocess.CalledProcessError:
                    print(f"  ❌ Failed to install {package}")
                    return False
        
        return True
    
    def start_mechanic_engine(self):
        """Start the main mechanic engine in background"""
        try:
            # Add to Python path
            if str(self.mechanic_path) not in sys.path:
                sys.path.insert(0, str(self.mechanic_path))
            
            # Import and start mechanic
            from bot_mechanic import LocalBotMechanic
            
            self.mechanic = LocalBotMechanic()
            
            # Start background monitoring
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            print("✅ Mechanic engine started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start mechanic: {e}")
            return False
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        print("🔄 Starting background monitoring...")
        
        while self.running:
            try:
                # Run quick health check
                if hasattr(self, 'mechanic'):
                    health_status = self.mechanic.quick_scan()
                    
                    # Auto-fix issues if found
                    if not health_status['healthy']:
                        print(f"\n🔧 Auto-fixing {health_status['issues']} issues...")
                        self.mechanic.auto_fix_all()
                        
                        # Run another check to verify fixes
                        post_fix_status = self.mechanic.quick_scan()
                        if post_fix_status['healthy']:
                            print("✅ All issues resolved")
                        else:
                            print(f"⚠️ {post_fix_status['issues']} issues remain")
                    
                    # Update dashboard data every check
                    self.update_dashboard_status(health_status)
                
                # Wait for next check
                for _ in range(self.check_interval):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ Monitor error: {e}")
                time.sleep(10)  # Wait before retrying
                
        print("🛑 Background monitoring stopped")
    
    def update_dashboard_status(self, health_status):
        """Update dashboard with current status"""
        try:
            status_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'healthy': health_status['healthy'],
                'issues_count': health_status['issues'],
                'health_score': health_status.get('health_score', 100),
                'services': health_status.get('checks', {}),
                'version': self.version
            }
            
            # Save status for dashboard to read
            status_file = self.base_path / "Intelligence" / "mechanic" / "database" / "dashboard_status.json"
            status_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Failed to update dashboard status: {e}")
            print(f"Monitor error: {e}")
            time.sleep(30)
    
    def create_flask_dashboard(self):
        """Create integrated Flask dashboard"""
        try:
            from flask import Flask, render_template_string, jsonify, request
            self.flask_app = Flask(__name__)
            
            @self.flask_app.route('/')
            def dashboard():
                return render_template_string(DASHBOARD_TEMPLATE)
            
            @self.flask_app.route('/mechanic/dashboard')
            def mechanic_dashboard():
                """Mechanic dashboard for iframe integration"""
                return render_template_string(MECHANIC_DASHBOARD_TEMPLATE)
            
            @self.flask_app.route('/api/status')
            def api_status():
                """Legacy API endpoint"""
                return mechanic_api_status()
            
            @self.flask_app.route('/mechanic/api/status')
            def mechanic_api_status():
                """API endpoint for mechanic status"""
                try:
                    if hasattr(self, 'mechanic'):
                        # Get health status
                        health = self.mechanic.quick_scan()
                        
                        # Get knowledge data
                        knowledge = self.mechanic.knowledge
                        
                        # Calculate stats
                        total_files = len(knowledge.get('files', {}))
                        total_functions = 0
                        categories = {}
                        
                        for file_data in knowledge.get('files', {}).values():
                            functions = file_data.get('features', {}).get('functions', [])
                            total_functions += len(functions)
                            
                            for func in functions:
                                cat = func.get('category', 'unknown')
                                categories[cat] = categories.get(cat, 0) + 1
                        
                        # Recent repairs
                        recent_repairs = self.mechanic.repair_history[-5:] if self.mechanic.repair_history else []
                        
                        return jsonify({
                            'status': 'healthy' if health['healthy'] else 'issues',
                            'issues_count': health['issues'],
                            'total_files': total_files,
                            'total_functions': total_functions,
                            'categories': categories,
                            'recent_repairs': recent_repairs,
                            'last_scan': knowledge.get('last_scan', {}).get('timestamp'),
                            'uptime': str(datetime.utcnow() - self.mechanic.start_time)
                        })
                    else:
                        return jsonify({'status': 'starting', 'message': 'Mechanic engine starting...'})
                        
                except Exception as e:
                    return jsonify({'status': 'error', 'message': str(e)})
            
            @self.flask_app.route('/api/scan', methods=['POST'])
            def api_scan():
                """Trigger full scan"""
                try:
                    if hasattr(self, 'mechanic'):
                        results = self.mechanic.deep_scan(verbose=False)
                        return jsonify({'success': True, 'results': results})
                    else:
                        return jsonify({'success': False, 'message': 'Mechanic not ready'})
                except Exception as e:
                    return jsonify({'success': False, 'message': str(e)})
            
            @self.flask_app.route('/api/fix', methods=['POST'])
            def api_fix():
                """Trigger auto-fix"""
                try:
                    if hasattr(self, 'mechanic'):
                        self.mechanic.auto_fix_all()
                        return jsonify({'success': True, 'message': 'Auto-fix completed'})
                    else:
                        return jsonify({'success': False, 'message': 'Mechanic not ready'})
                except Exception as e:
                    return jsonify({'success': False, 'message': str(e)})
            
            return True
        except ImportError:
            print("⚠️ Flask not available, dashboard disabled")
            return False
    
    def start_dashboard(self):
        """Start the Flask dashboard"""
        try:
            if not self.create_flask_dashboard():
                return False
            
            def run_flask():
                import logging
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                
                self.flask_app.run(
                    host='127.0.0.1',
                    port=self.dashboard_port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                )
            
            self.flask_thread = threading.Thread(target=run_flask, daemon=True)
            self.flask_thread.start()
            
            print(f"🌐 Dashboard started on {self.config.get_dashboard_url()}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start dashboard: {e}")
            return False
    
    def integrate_with_main_dashboard(self):
        """Integrate with main bot dashboard"""
        try:
            main_dashboard_path = self.base_path / 'wwwroot' / 'unified-dashboard.html'
            
            if main_dashboard_path.exists():
                # Update main dashboard to include mechanic tab
                content = main_dashboard_path.read_text()
                
                # Check if mechanic tab already exists
                if 'bot-mechanic-tab' not in content:
                    # Add mechanic tab to navigation
                    nav_addition = '''
                    <button class="tab-button" onclick="openTab(event, 'bot-mechanic-tab')">
                        🧠 Bot Mechanic
                    </button>'''
                    
                    content = content.replace(
                        '<button class="tab-button" onclick="openTab(event, \'monitoring-tab\')">',
                        nav_addition + '\n                    <button class="tab-button" onclick="openTab(event, \'monitoring-tab\')">'
                    )
                    
                    # Add mechanic tab content
                    tab_content = f'''
                    <div id="bot-mechanic-tab" class="tab-content">
                        <h2>🧠 Bot Mechanic - Local System Monitor</h2>
                        <div class="status-grid">
                            <div class="status-card">
                                <h3>🔍 System Status</h3>
                                <iframe src="{self.config.get_iframe_src()}" 
                                        width="100%" height="600px" frameborder="0">
                                </iframe>
                            </div>
                        </div>
                    </div>'''
                    
                    content = content.replace(
                        '</div>\n    </div>\n\n    <script>',
                        tab_content + '\n        </div>\n    </div>\n\n    <script>'
                    )
                    
                    main_dashboard_path.write_text(content)
                    print("✅ Integrated with main dashboard")
                else:
                    print("✅ Already integrated with main dashboard")
            else:
                print("⚠️ Main dashboard not found, running standalone")
                
        except Exception as e:
            print(f"⚠️ Dashboard integration failed: {e}")
    
    def start(self):
        """Start the complete system"""
        print("\n" + "="*60)
        print("🚀 STARTING LOCAL BOT MECHANIC AUTO-LAUNCHER")
        print("="*60)
        
        # Check dependencies
        if not self.check_dependencies():
            print("❌ Dependency check failed")
            return False
        
        # Start mechanic engine
        print("\n🧠 Starting mechanic engine...")
        if not self.start_mechanic_engine():
            print("❌ Failed to start mechanic engine")
            return False
        
        # Start dashboard
        print("\n🌐 Starting dashboard...")
        if not self.start_dashboard():
            print("⚠️ Dashboard failed, continuing without it")
        
        # Integrate with main dashboard
        print("\n🔗 Integrating with main dashboard...")
        self.integrate_with_main_dashboard()
        
        print("\n" + "="*60)
        print("✅ LOCAL BOT MECHANIC FULLY OPERATIONAL!")
        print("="*60)
        print(f"🌐 Dashboard: {self.config.get_dashboard_url()}")
        print(f"🔍 Monitoring every {self.check_interval} seconds")
        print(f"🔧 Auto-repair enabled")
        print("="*60)
        
        return True
    
    def run_forever(self):
        """Keep running until stopped"""
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        """Stop all services"""
        print("\n🛑 Stopping Local Bot Mechanic...")
        self.running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            print("  • Stopping monitor...")
        
        if self.flask_thread and self.flask_thread.is_alive():
            print("  • Stopping dashboard...")
        
        print("✅ Stopped gracefully")

# Dashboard HTML Template
DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local Bot Mechanic Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white; 
            min-height: 100vh;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .header h1 { 
            font-size: 2.5em; 
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .status-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 20px; 
            margin-bottom: 30px;
        }
        .status-card { 
            background: rgba(255,255,255,0.15); 
            padding: 20px; 
            border-radius: 15px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        .status-card:hover {
            transform: translateY(-5px);
        }
        .status-card h3 { 
            margin-bottom: 15px; 
            color: #ffd700;
            font-size: 1.3em;
        }
        .metric { 
            display: flex; 
            justify-content: space-between; 
            margin: 10px 0; 
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric:last-child { border-bottom: none; }
        .metric-value { 
            font-weight: bold; 
            color: #4ade80; 
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-healthy { background-color: #4ade80; }
        .status-warning { background-color: #fbbf24; }
        .status-error { background-color: #ef4444; }
        .action-buttons {
            display: flex;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s ease;
            background: linear-gradient(45deg, #4ade80, #22d3ee);
            color: white;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .btn-secondary {
            background: linear-gradient(45deg, #8b5cf6, #a855f7);
        }
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .feature-item {
            background: rgba(255,255,255,0.1);
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.9em;
        }
        .loading {
            text-align: center;
            font-style: italic;
            opacity: 0.7;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .pulse { animation: pulse 2s infinite; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Local Bot Mechanic</h1>
            <p>Intelligent System Monitor & Auto-Repair</p>
            <div class="action-buttons">
                <button class="btn" onclick="triggerScan()">🔍 Full Scan</button>
                <button class="btn btn-secondary" onclick="triggerFix()">🔧 Auto Fix</button>
                <button class="btn btn-secondary" onclick="refreshData()">🔄 Refresh</button>
            </div>
        </div>

        <div class="status-grid">
            <div class="status-card">
                <h3>📊 System Overview</h3>
                <div id="overview-content" class="loading">Loading...</div>
            </div>

            <div class="status-card">
                <h3>📁 Code Analysis</h3>
                <div id="analysis-content" class="loading">Loading...</div>
            </div>

            <div class="status-card">
                <h3>🔧 Recent Repairs</h3>
                <div id="repairs-content" class="loading">Loading...</div>
            </div>

            <div class="status-card">
                <h3>⚡ Features Detected</h3>
                <div id="features-content" class="loading">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        let statusData = {};

        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                statusData = await response.json();
                updateDisplay();
            } catch (error) {
                console.error('Error fetching status:', error);
                showError('Failed to fetch status');
            }
        }

        function updateDisplay() {
            updateOverview();
            updateAnalysis();
            updateRepairs();
            updateFeatures();
        }

        function updateOverview() {
            const content = document.getElementById('overview-content');
            const status = statusData.status || 'unknown';
            const statusClass = status === 'healthy' ? 'status-healthy' : 
                               status === 'issues' ? 'status-warning' : 'status-error';
            
            content.innerHTML = `
                <div class="metric">
                    <span>Status:</span>
                    <span><span class="status-indicator ${statusClass}"></span>${status.toUpperCase()}</span>
                </div>
                <div class="metric">
                    <span>Issues Found:</span>
                    <span class="metric-value">${statusData.issues_count || 0}</span>
                </div>
                <div class="metric">
                    <span>Uptime:</span>
                    <span class="metric-value">${statusData.uptime || 'N/A'}</span>
                </div>
                <div class="metric">
                    <span>Last Scan:</span>
                    <span class="metric-value">${formatTime(statusData.last_scan) || 'Never'}</span>
                </div>
            `;
        }

        function updateAnalysis() {
            const content = document.getElementById('analysis-content');
            content.innerHTML = `
                <div class="metric">
                    <span>Files Tracked:</span>
                    <span class="metric-value">${statusData.total_files || 0}</span>
                </div>
                <div class="metric">
                    <span>Functions Found:</span>
                    <span class="metric-value">${statusData.total_functions || 0}</span>
                </div>
            `;
        }

        function updateRepairs() {
            const content = document.getElementById('repairs-content');
            const repairs = statusData.recent_repairs || [];
            
            if (repairs.length === 0) {
                content.innerHTML = '<div class="loading">No recent repairs</div>';
                return;
            }

            let html = '';
            repairs.slice(-5).forEach(repair => {
                const status = repair.success ? '✅' : '❌';
                const time = formatTime(repair.timestamp);
                html += `
                    <div class="metric">
                        <span>${status} ${repair.system}</span>
                        <span style="font-size: 0.8em; opacity: 0.7;">${time}</span>
                    </div>
                `;
            });
            content.innerHTML = html;
        }

        function updateFeatures() {
            const content = document.getElementById('features-content');
            const categories = statusData.categories || {};
            
            if (Object.keys(categories).length === 0) {
                content.innerHTML = '<div class="loading">No features detected</div>';
                return;
            }

            const sorted = Object.entries(categories)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 8);

            let html = '<div class="features-grid">';
            sorted.forEach(([category, count]) => {
                html += `
                    <div class="feature-item">
                        <strong>${category}</strong><br>
                        <span style="opacity: 0.8;">${count} items</span>
                    </div>
                `;
            });
            html += '</div>';
            content.innerHTML = html;
        }

        async function triggerScan() {
            showLoading('Running full scan...');
            try {
                const response = await fetch('/api/scan', { method: 'POST' });
                const result = await response.json();
                if (result.success) {
                    await fetchStatus();
                    showSuccess('Scan completed successfully');
                } else {
                    showError(result.message || 'Scan failed');
                }
            } catch (error) {
                showError('Failed to trigger scan');
            }
        }

        async function triggerFix() {
            showLoading('Running auto-fix...');
            try {
                const response = await fetch('/api/fix', { method: 'POST' });
                const result = await response.json();
                if (result.success) {
                    await fetchStatus();
                    showSuccess('Auto-fix completed');
                } else {
                    showError(result.message || 'Fix failed');
                }
            } catch (error) {
                showError('Failed to trigger fix');
            }
        }

        function refreshData() {
            fetchStatus();
            showSuccess('Data refreshed');
        }

        function formatTime(isoString) {
            if (!isoString) return null;
            try {
                return new Date(isoString).toLocaleString();
            } catch {
                return isoString;
            }
        }

        function showLoading(message) {
            console.log('Loading:', message);
        }

        function showSuccess(message) {
            console.log('Success:', message);
        }

        function showError(message) {
            console.error('Error:', message);
        }

        // Auto-refresh every 30 seconds
        setInterval(fetchStatus, 30000);

        // Initial load
        fetchStatus();
    </script>
</body>
</html>
'''

# Mechanic-specific Dashboard Template for iframe integration
MECHANIC_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bot Mechanic - Embedded</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e6ed; 
            min-height: 100vh;
            padding: 15px;
        }
        .mini-container { 
            max-width: 100%; 
            margin: 0 auto; 
        }
        .mini-header {
            text-align: center;
            margin-bottom: 20px;
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            backdrop-filter: blur(10px);
        }
        .mini-header h2 { 
            font-size: 1.5em; 
            margin-bottom: 5px;
            color: #4CAF50;
        }
        .mini-status-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 15px; 
            margin-bottom: 20px;
        }
        .mini-status-card { 
            background: rgba(255,255,255,0.1); 
            padding: 15px; 
            border-radius: 10px; 
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .mini-status-card h4 { 
            margin-bottom: 10px; 
            color: #79c0ff;
            font-size: 0.9em;
        }
        .mini-metric { 
            display: flex; 
            justify-content: space-between; 
            margin-bottom: 5px;
            font-size: 0.8em;
        }
        .mini-metric-value { 
            font-weight: bold; 
            color: #4CAF50; 
        }
        .mini-refresh-btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.8em;
            width: 100%;
            margin-top: 10px;
        }
        .mini-refresh-btn:hover {
            background: #45a049;
        }
        .mini-recent-list {
            max-height: 120px;
            overflow-y: auto;
            background: rgba(0,0,0,0.3);
            border-radius: 5px;
            padding: 8px;
        }
        .mini-recent-item {
            font-size: 0.75em;
            padding: 3px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .mini-recent-item:last-child {
            border-bottom: none;
        }
    </style>
</head>
<body>
    <div class="mini-container">
        <div class="mini-header">
            <h2>🔧 Bot Mechanic</h2>
            <div id="mini-last-update">Last update: --:--:--</div>
        </div>

        <div class="mini-status-grid">
            <div class="mini-status-card">
                <h4>📊 Health Overview</h4>
                <div class="mini-metric">
                    <span>Health Score:</span>
                    <span class="mini-metric-value" id="mini-health-score">--</span>
                </div>
                <div class="mini-metric">
                    <span>Active Issues:</span>
                    <span class="mini-metric-value" id="mini-issues-count">--</span>
                </div>
                <div class="mini-metric">
                    <span>Files Tracked:</span>
                    <span class="mini-metric-value" id="mini-files-count">--</span>
                </div>
                <div class="mini-metric">
                    <span>Features Found:</span>
                    <span class="mini-metric-value" id="mini-features-count">--</span>
                </div>
            </div>

            <div class="mini-status-card">
                <h4>🔧 Recent Activity</h4>
                <div class="mini-recent-list" id="mini-recent-repairs">
                    <div class="mini-recent-item">Loading recent repairs...</div>
                </div>
            </div>
        </div>

        <button class="mini-refresh-btn" onclick="fetchMiniStatus()">🔄 Refresh Status</button>
    </div>

    <script>
        async function fetchMiniStatus() {
            try {
                const response = await fetch('/mechanic/api/status');
                if (response.ok) {
                    const data = await response.json();
                    
                    // Update health metrics
                    document.getElementById('mini-health-score').textContent = data.health_score + '%';
                    document.getElementById('mini-issues-count').textContent = data.issues_count;
                    document.getElementById('mini-files-count').textContent = data.files_count || '--';
                    document.getElementById('mini-features-count').textContent = data.feature_count || '--';
                    
                    // Update recent repairs
                    const repairsContainer = document.getElementById('mini-recent-repairs');
                    if (data.recent_repairs && data.recent_repairs.length > 0) {
                        repairsContainer.innerHTML = data.recent_repairs.map(repair => 
                            `<div class="mini-recent-item">${repair.system}: ${repair.success ? '✅' : '❌'}</div>`
                        ).join('');
                    } else {
                        repairsContainer.innerHTML = '<div class="mini-recent-item">No recent repairs</div>';
                    }
                    
                    // Update timestamp
                    document.getElementById('mini-last-update').textContent = 
                        'Last update: ' + new Date().toLocaleTimeString();
                }
            } catch (error) {
                console.error('Failed to fetch status:', error);
                document.getElementById('mini-health-score').textContent = 'Error';
                document.getElementById('mini-issues-count').textContent = '--';
            }
        }

        // Auto-refresh every 15 seconds for embedded view
        setInterval(fetchMiniStatus, 15000);

        // Initial load
        fetchMiniStatus();
    </script>
</body>
</html>
'''

def main():
    """Main entry point"""
    launcher = MechanicAutoLauncher()
    
    if launcher.start():
        print("\n🎯 Auto-launcher ready! The mechanic will:")
        print("   • Monitor your bot continuously")
        print("   • Auto-fix issues as they appear")
        print("   • Integrate with your main dashboard")
        print("   • Run completely locally")
        print("\nPress Ctrl+C to stop\n")
        
        launcher.run_forever()
    else:
        print("❌ Failed to start auto-launcher")
        sys.exit(1)

if __name__ == "__main__":
    main()
