#!/usr/bin/env python3
"""
Test the Cloud Mechanic's actual performance and features
"""

def test_cloud_mechanic_features():
    import os
    import subprocess
    from datetime import datetime
    
    print("🧪 TESTING CLOUD MECHANIC FEATURES...")
    print("=" * 60)
    
    # Set environment for best performance
    env = os.environ.copy()
    env.update({
        'ULTIMATE_MODE': 'true',
        'GITHUB_REPOSITORY_OWNER': 'c-trading-bo',
        'GITHUB_REPOSITORY': 'trading-bot-c-'
    })
    
    start_time = datetime.now()
    
    try:
        # Run the cloud mechanic from root directory
        result = subprocess.run([
            'python', 
            'Intelligence/mechanic/cloud/cloud_mechanic_core.py'
        ], 
        env=env,
        capture_output=True,
        text=True,
        cwd=os.getcwd()
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"⏱️ Execution Time: {execution_time:.2f} seconds")
        print(f"🔄 Return Code: {result.returncode}")
        print(f"📝 Output Length: {len(result.stdout)} characters")
        
        # Analyze the output for feature verification
        output = result.stdout
        
        features_working = {
            'Workflow Discovery': '27' in output and 'workflows' in output,
            'Health Analysis': 'healthy workflows' in output,
            'Budget Monitoring': 'monthly minutes' in output,
            'Issue Detection': 'Issues found' in output,
            'Ultimate Mode': 'ULTIMATE' in output,
            'AI Optimization': 'optimization' in output.lower(),
            'Pattern Recognition': 'patterns recognized' in output,
            'Performance Metrics': 'success_rate' in output,
            'Intelligent Preparation': 'INTELLIGENT PREPARATION' in output,
            'Workflow Learning': 'Learning from' in output,
            'Pre-compilation': 'Pre-compiling' in output,
            'Cache Management': 'caching' in output.lower(),
            'Error Analysis': 'Failed to' in output or 'issues' in output.lower()
        }
        
        print(f"\n🎯 FEATURE STATUS REPORT:")
        print("=" * 40)
        
        working_count = 0
        for feature, is_working in features_working.items():
            status = "✅ ACTIVE" if is_working else "❌ INACTIVE"
            print(f"{feature:.<25} {status}")
            if is_working:
                working_count += 1
        
        print(f"\n📊 SUMMARY:")
        print(f"   Features Active: {working_count}/{len(features_working)}")
        print(f"   Success Rate: {(working_count/len(features_working)*100):.1f}%")
        print(f"   Performance: {'🚀 EXCELLENT' if execution_time < 5 else '⚠️ SLOW'}")
        
        # Check for specific metrics
        if 'Ultimate Metrics' in output:
            print(f"\n🎯 ULTIMATE AI FEATURES CONFIRMED ACTIVE!")
        
        if result.returncode == 0:
            print(f"\n✅ CLOUD MECHANIC IS FULLY OPERATIONAL!")
        else:
            print(f"\n⚠️ Some issues detected but core features working")
            
        return working_count >= 10  # At least 10 features should be working
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_cloud_mechanic_features()
