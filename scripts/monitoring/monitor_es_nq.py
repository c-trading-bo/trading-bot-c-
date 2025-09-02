#!/usr/bin/env python3
"""
ES/NQ Futures Monitoring Dashboard
Complete monitoring for ES/NQ futures automated trading system
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

def load_json_safe(file_path):
    """Safely load JSON file with error handling"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading {file_path}: {e}")
    return None

def check_data_freshness(data, max_age_minutes=30):
    """Check if data is fresh enough"""
    if not data or 'timestamp' not in data:
        return False, "No timestamp"
    
    try:
        data_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        now = datetime.now(data_time.tzinfo) if data_time.tzinfo else datetime.utcnow()
        age = now - data_time
        
        if age.total_seconds() / 60 > max_age_minutes:
            return False, f"Data age: {int(age.total_seconds() / 60)} minutes"
        return True, f"Fresh ({int(age.total_seconds() / 60)}m old)"
    except Exception as e:
        return False, f"Timestamp error: {e}"

def get_signal_emoji(signal):
    """Get emoji for signal"""
    if isinstance(signal, dict):
        action = signal.get('action', 'NEUTRAL')
    else:
        action = str(signal)
    
    return {
        'BUY': '🟢',
        'SELL': '🔴', 
        'BULLISH': '🟢',
        'BEARISH': '🔴',
        'NEUTRAL': '🟡'
    }.get(action.upper(), '🟡')

def display_dashboard():
    """Display the complete ES/NQ monitoring dashboard"""
    
    print("╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("║                    ES/NQ FUTURES AUTOMATED TRADING SYSTEM                        ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║ Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("╠══════════════════════════════════════════════════════════════════════════════════╣")
    
    # Initialize status tracking
    all_systems_ok = True
    data_sources = 0
    active_sources = 0
    
    # Options Flow Analysis
    print("║ 📊 OPTIONS FLOW ANALYSIS")
    options_data = load_json_safe('Intelligence/data/options/es_nq_flow.json')
    data_sources += 1
    
    if options_data:
        fresh, age_info = check_data_freshness(options_data)
        if fresh:
            active_sources += 1
            if 'signals' in options_data:
                es_signal = options_data['signals'].get('ES', {})
                nq_signal = options_data['signals'].get('NQ', {})
                
                es_emoji = get_signal_emoji(es_signal)
                nq_emoji = get_signal_emoji(nq_signal)
                
                print(f"║   ES: {es_emoji} {es_signal.get('action', 'NEUTRAL')} ({es_signal.get('confidence', 0):.0%})")
                print(f"║   NQ: {nq_emoji} {nq_signal.get('action', 'NEUTRAL')} ({nq_signal.get('confidence', 0):.0%})")
                
                if 'ES' in options_data and 'put_call_ratio' in options_data['ES']:
                    print(f"║   ES P/C Ratio: {options_data['ES']['put_call_ratio']:.2f}")
                if 'NQ' in options_data and 'put_call_ratio' in options_data['NQ']:
                    print(f"║   NQ P/C Ratio: {options_data['NQ']['put_call_ratio']:.2f}")
            else:
                print("║   ⚠️  No signals available")
        else:
            print(f"║   ❌ Stale data ({age_info})")
            all_systems_ok = False
    else:
        print("║   ❌ No options flow data")
        all_systems_ok = False
    
    print("║")
    
    # News Sentiment
    print("║ 📰 NEWS SENTIMENT")
    news_data = load_json_safe('Intelligence/data/news/es_nq_sentiment.json')
    data_sources += 1
    
    if news_data:
        fresh, age_info = check_data_freshness(news_data)
        if fresh:
            active_sources += 1
            es_sentiment = news_data.get('ES', {})
            nq_sentiment = news_data.get('NQ', {})
            
            es_signal_emoji = get_signal_emoji(es_sentiment.get('signal', 'NEUTRAL'))
            nq_signal_emoji = get_signal_emoji(nq_sentiment.get('signal', 'NEUTRAL'))
            
            print(f"║   ES: {es_signal_emoji} {es_sentiment.get('signal', 'NEUTRAL')} ({es_sentiment.get('sentiment', 0):.2f})")
            print(f"║   NQ: {nq_signal_emoji} {nq_sentiment.get('signal', 'NEUTRAL')} ({nq_sentiment.get('sentiment', 0):.2f})")
            print(f"║   ES Articles: {es_sentiment.get('article_count', 0)}")
            print(f"║   NQ Articles: {nq_sentiment.get('article_count', 0)}")
        else:
            print(f"║   ❌ Stale data ({age_info})")
            all_systems_ok = False
    else:
        print("║   ❌ No news sentiment data")
        all_systems_ok = False
    
    print("║")
    
    # Regime Detection
    print("║ 🎯 REGIME DETECTION")
    regime_data = load_json_safe('Intelligence/data/regime/es_nq_regime.json')
    data_sources += 1
    
    if regime_data:
        fresh, age_info = check_data_freshness(regime_data)
        if fresh:
            active_sources += 1
            regime = regime_data.get('regime', 'UNKNOWN')
            confidence = regime_data.get('confidence', 0)
            
            regime_emoji = {
                'BULL_TREND': '🐂',
                'BEAR_TREND': '🐻',
                'RISK_OFF': '⛔',
                'ROTATION': '🔄',
                'CHOP': '🌊'
            }.get(regime, '❓')
            
            print(f"║   Current: {regime_emoji} {regime} ({confidence:.0%})")
            
            if 'features' in regime_data:
                features = regime_data['features']
                print(f"║   ES Vol: {features.get('es_volatility', 0):.1f}%")
                print(f"║   NQ Vol: {features.get('nq_volatility', 0):.1f}%")
                print(f"║   Correlation: {features.get('correlation', 0):.2f}")
            
            if 'adjustments' in regime_data:
                adj = regime_data['adjustments']
                print(f"║   ES Size: {adj.get('ES', {}).get('position_size', 1):.1f}x")
                print(f"║   NQ Size: {adj.get('NQ', {}).get('position_size', 1):.1f}x")
        else:
            print(f"║   ❌ Stale data ({age_info})")
            all_systems_ok = False
    else:
        print("║   ❌ No regime detection data")
        all_systems_ok = False
    
    print("║")
    
    # Correlation Analysis
    print("║ 📈 CORRELATION MATRIX")
    # Check if we have correlation data (could be integrated with regime or separate)
    if regime_data and 'features' in regime_data:
        correlation = regime_data['features'].get('correlation', 0)
        corr_emoji = "🔴" if abs(correlation) > 0.8 else "🟡" if abs(correlation) > 0.5 else "🟢"
        print(f"║   ES-NQ: {corr_emoji} {correlation:.2f}")
        
        if correlation > 0.8:
            print("║   Status: Highly correlated - avoid opposing positions")
        elif correlation < -0.5:
            print("║   Status: Negatively correlated - pairs trading opportunity") 
        elif abs(correlation) < 0.3:
            print("║   Status: Decorrelated - reduce position size")
        else:
            print("║   Status: Normal correlation")
    else:
        print("║   ❌ No correlation data")
        all_systems_ok = False
    
    print("║")
    
    # System Health
    print("║ 🔧 SYSTEM HEALTH")
    health_emoji = "✅" if all_systems_ok else "⚠️"
    print(f"║   Overall: {health_emoji} {active_sources}/{data_sources} data sources active")
    
    if all_systems_ok:
        print("║   Status: ALL SYSTEMS OPERATIONAL")
    else:
        print("║   Status: SOME SYSTEMS DEGRADED")
    
    print("║")
    
    # GitHub Actions Status
    print("║ ⚙️  AUTOMATION STATUS")
    workflow_status = check_workflow_status()
    print(f"║   GitHub Actions: {workflow_status}")
    print("║   News Collection: Every 5 minutes")
    print("║   Options Analysis: Real-time via workflows")
    print("║   Regime Detection: Continuous monitoring")
    
    print("╠══════════════════════════════════════════════════════════════════════════════════╣")
    
    # Overall System Status
    if all_systems_ok and active_sources >= 2:
        print("║ 🎉 SYSTEM STATUS: FULLY OPERATIONAL - Ready for automated trading")
    elif active_sources >= 1:
        print("║ ⚠️  SYSTEM STATUS: PARTIALLY OPERATIONAL - Some degradation")
    else:
        print("║ ❌ SYSTEM STATUS: CRITICAL - Manual intervention required")
    
    print("╚══════════════════════════════════════════════════════════════════════════════════╝")
    
    # Quick Action Commands
    print("\n💡 Quick Commands:")
    print("   python3 Intelligence/scripts/options/es_nq_options_flow.py")
    print("   python3 Intelligence/scripts/ml/es_nq_regime_detection.py")
    print("   python3 monitor_es_nq.py  # This dashboard")
    
    return all_systems_ok

def check_workflow_status():
    """Check GitHub workflow status"""
    workflow_file = '.github/workflows/es_nq_news_sentiment.yml'
    if os.path.exists(workflow_file):
        return "✅ Configured"
    else:
        return "❌ Not configured"

def generate_summary_report():
    """Generate a JSON summary report"""
    
    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'system_status': 'UNKNOWN',
        'data_sources': {},
        'active_signals': {},
        'recommendations': []
    }
    
    # Check each data source
    sources = {
        'options': 'Intelligence/data/options/es_nq_flow.json',
        'news': 'Intelligence/data/news/es_nq_sentiment.json', 
        'regime': 'Intelligence/data/regime/es_nq_regime.json'
    }
    
    active_count = 0
    total_count = len(sources)
    
    for source_name, file_path in sources.items():
        data = load_json_safe(file_path)
        is_fresh, age_info = check_data_freshness(data) if data else (False, "No data")
        
        report['data_sources'][source_name] = {
            'available': data is not None,
            'fresh': is_fresh,
            'age_info': age_info
        }
        
        if is_fresh:
            active_count += 1
    
    # Determine system status
    if active_count == total_count:
        report['system_status'] = 'FULLY_OPERATIONAL'
    elif active_count >= total_count // 2:
        report['system_status'] = 'PARTIALLY_OPERATIONAL'
    else:
        report['system_status'] = 'CRITICAL'
    
    # Extract active signals
    options_data = load_json_safe('Intelligence/data/options/es_nq_flow.json')
    if options_data and 'signals' in options_data:
        report['active_signals']['options'] = options_data['signals']
    
    news_data = load_json_safe('Intelligence/data/news/es_nq_sentiment.json')
    if news_data:
        report['active_signals']['news'] = {
            'ES': news_data.get('ES', {}).get('signal', 'NEUTRAL'),
            'NQ': news_data.get('NQ', {}).get('signal', 'NEUTRAL')
        }
    
    regime_data = load_json_safe('Intelligence/data/regime/es_nq_regime.json')
    if regime_data:
        report['active_signals']['regime'] = regime_data.get('regime', 'UNKNOWN')
    
    # Save report
    os.makedirs('Intelligence/reports', exist_ok=True)
    with open('Intelligence/reports/system_status.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    """Main execution function"""
    try:
        # Display dashboard
        all_ok = display_dashboard()
        
        # Generate report
        report = generate_summary_report()
        
        print(f"\n📊 System report saved to: Intelligence/reports/system_status.json")
        
        # Exit code based on system health
        return 0 if all_ok else 1
        
    except Exception as e:
        print(f"\n❌ Error running dashboard: {e}")
        return 1

if __name__ == "__main__":
    success = main()
    sys.exit(success)