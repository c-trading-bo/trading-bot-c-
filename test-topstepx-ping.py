#!/usr/bin/env python3
"""
Quick TopstepX connectivity test
"""
import requests
import time
import json

def test_topstepx_endpoints():
    """Test all TopstepX endpoints for connectivity"""
    
    endpoints = {
        "REST API": "https://api.topstepx.com",
        "User Hub": "https://rtc.topstepx.com/hubs/user", 
        "Market Hub": "https://rtc.topstepx.com/hubs/market"
    }
    
    print("🔍 Testing TopstepX endpoint connectivity...")
    print("=" * 50)
    
    results = {}
    
    for name, url in endpoints.items():
        try:
            print(f"Testing {name}: {url}")
            
            # Basic connectivity test
            start_time = time.time()
            response = requests.get(url, timeout=10)
            latency = (time.time() - start_time) * 1000
            
            status = "✅ REACHABLE" if response.status_code < 500 else "⚠️ SERVER ERROR"
            results[name] = {
                "status": status,
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
                "headers": dict(response.headers)
            }
            
            print(f"  Status: {status} ({response.status_code})")
            print(f"  Latency: {latency:.1f}ms")
            
        except requests.exceptions.ConnectTimeout:
            print(f"  Status: ❌ TIMEOUT")
            results[name] = {"status": "❌ TIMEOUT", "error": "Connection timeout"}
            
        except requests.exceptions.ConnectionError as e:
            print(f"  Status: ❌ CONNECTION ERROR")
            results[name] = {"status": "❌ CONNECTION ERROR", "error": str(e)}
            
        except Exception as e:
            print(f"  Status: ❌ ERROR: {e}")
            results[name] = {"status": "❌ ERROR", "error": str(e)}
            
        print()
    
    # Summary
    print("📊 SUMMARY:")
    print("=" * 50)
    
    for name, result in results.items():
        status = result.get("status", "❌ UNKNOWN")
        latency = result.get("latency_ms", "N/A")
        print(f"{name}: {status} ({latency}ms)" if latency != "N/A" else f"{name}: {status}")
    
    # Check if any are working
    working = [name for name, result in results.items() if "✅" in result.get("status", "")]
    if working:
        print(f"\n🎉 {len(working)}/{len(endpoints)} endpoints reachable!")
        print(f"Working: {', '.join(working)}")
    else:
        print(f"\n⚠️ No endpoints reachable - check internet connection or TopstepX status")
    
    return results

if __name__ == "__main__":
    test_topstepx_endpoints()
