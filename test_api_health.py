#!/usr/bin/env python3
"""
Test API endpoints used by the workflows to ensure they're working correctly
"""

import requests
import yfinance as yf
import feedparser
from datetime import datetime
import json

def test_yahoo_finance_api():
    """Test Yahoo Finance API"""
    try:
        print("🧪 Testing Yahoo Finance API...")
        ticker = yf.Ticker("SPY")
        data = ticker.history(period="1d")
        
        if not data.empty:
            latest_price = data['Close'].iloc[-1]
            print(f"✅ Yahoo Finance API: Working (SPY: ${latest_price:.2f})")
            return True, latest_price
        else:
            print("❌ Yahoo Finance API: No data returned")
            return False, None
            
    except Exception as e:
        print(f"❌ Yahoo Finance API: Error - {e}")
        return False, None

def test_rss_feeds():
    """Test RSS feed endpoints"""
    rss_feeds = [
        "https://finance.yahoo.com/rss/topfinstories",
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
    ]
    
    working_feeds = 0
    total_articles = 0
    
    print("🧪 Testing RSS Feeds...")
    
    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            if len(feed.entries) > 0:
                working_feeds += 1
                total_articles += len(feed.entries)
                print(f"✅ RSS Feed: {feed_url[:50]}... ({len(feed.entries)} articles)")
            else:
                print(f"❌ RSS Feed: {feed_url[:50]}... (No articles)")
        except Exception as e:
            print(f"❌ RSS Feed: {feed_url[:50]}... (Error: {e})")
    
    success = working_feeds > 0
    print(f"📊 RSS Summary: {working_feeds}/{len(rss_feeds)} feeds working, {total_articles} total articles")
    return success, working_feeds

def test_http_requests():
    """Test basic HTTP connectivity"""
    try:
        print("🧪 Testing HTTP connectivity...")
        response = requests.get("https://httpbin.org/status/200", timeout=10)
        
        if response.status_code == 200:
            print("✅ HTTP Requests: Working")
            return True
        else:
            print(f"❌ HTTP Requests: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ HTTP Requests: Error - {e}")
        return False

def test_workflow_data_integration():
    """Test if the BotCore integration script exists and is functional"""
    import os
    
    integration_script = "Intelligence/scripts/workflow_data_integration.py"
    
    print("🧪 Testing BotCore Integration Script...")
    
    if os.path.exists(integration_script):
        print(f"✅ Integration Script: Found at {integration_script}")
        
        # Test if it can be imported
        try:
            # Create a simple test to see if the script runs
            import subprocess
            result = subprocess.run(['python3', integration_script, '--help'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 or "usage" in result.stdout.lower() or "help" in result.stdout.lower():
                print("✅ Integration Script: Executable and responds to --help")
                return True
            else:
                print(f"⚠️  Integration Script: May have issues (exit code: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"⚠️  Integration Script: Cannot test execution - {e}")
            return False
    else:
        print(f"❌ Integration Script: Not found at {integration_script}")
        return False

def main():
    """Run comprehensive API and integration tests"""
    print("🔍 COMPREHENSIVE API & INTEGRATION HEALTH CHECK")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    test_results = {}
    
    # Test Yahoo Finance API
    yf_success, yf_price = test_yahoo_finance_api()
    test_results['yahoo_finance'] = {
        'status': 'OK' if yf_success else 'FAILED',
        'latest_spy_price': yf_price
    }
    
    print()
    
    # Test RSS Feeds
    rss_success, rss_count = test_rss_feeds()
    test_results['rss_feeds'] = {
        'status': 'OK' if rss_success else 'FAILED',
        'working_feeds': rss_count
    }
    
    print()
    
    # Test HTTP connectivity
    http_success = test_http_requests()
    test_results['http_connectivity'] = {
        'status': 'OK' if http_success else 'FAILED'
    }
    
    print()
    
    # Test BotCore integration
    integration_success = test_workflow_data_integration()
    test_results['botcore_integration'] = {
        'status': 'OK' if integration_success else 'NEEDS_ATTENTION'
    }
    
    print()
    print("=" * 60)
    print("📊 API HEALTH SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = len([r for r in test_results.values() if r['status'] == 'OK'])
    
    for component, result in test_results.items():
        status_icon = "✅" if result['status'] == 'OK' else "❌" if result['status'] == 'FAILED' else "⚠️ "
        print(f"{status_icon} {component.replace('_', ' ').title()}: {result['status']}")
    
    print(f"\n🎯 Overall API Health: {passed_tests}/{total_tests} components working")
    
    if passed_tests == total_tests:
        print("🎉 ALL APIs and integrations are working correctly!")
        print("🚀 Your trading bot has full data access and integration capability!")
    elif passed_tests >= total_tests - 1:
        print("👍 Most APIs are working correctly - minor issues detected")
        print("✨ Your trading bot should function well with current setup")
    else:
        print("⚠️  Multiple API issues detected - troubleshooting needed")
        print("🔧 Some workflows may not function optimally")
    
    # Save results
    test_results['test_timestamp'] = datetime.now().isoformat()
    test_results['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'overall_health': 'EXCELLENT' if passed_tests == total_tests else 'GOOD' if passed_tests >= total_tests - 1 else 'NEEDS_ATTENTION'
    }
    
    with open('api_health_check_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: api_health_check_results.json")

if __name__ == "__main__":
    main()