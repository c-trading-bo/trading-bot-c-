#!/usr/bin/env python3
"""
Demo script to show Ultimate Cloud Mechanic features
"""

import os
import sys
import asyncio
from pathlib import Path

# Set environment variables for testing
os.environ['GITHUB_REPOSITORY_OWNER'] = 'c-trading-bo'
os.environ['GITHUB_REPOSITORY'] = 'c-trading-bo/trading-bot-c-'
os.environ['ULTIMATE_MODE'] = 'true'

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def demo_ultimate_features():
    """Demonstrate the ultimate features"""
    print("🚀 ULTIMATE CLOUD MECHANIC DEMO")
    print("=" * 50)
    
    # Import after setting environment
    from cloud_mechanic_core import CloudMechanicUltimate
    
    # Create ultimate mechanic
    mechanic = CloudMechanicUltimate()
    
    print("\n1. 🧠 Learning from workflows...")
    repo_info = await mechanic.learn_all_workflows()
    
    print(f"   • Found {len(repo_info.get('workflows', []))} workflows")
    for workflow in repo_info.get('workflows', []):
        print(f"     - {workflow.get('name', 'Unknown')}: {len(workflow.get('optimizations', []))} optimizations")
    
    print("\n2. ⚡ Intelligent workflow preparation...")
    preparation = await mechanic.prepare_workflow_intelligent(repo_info)
    
    print(f"   • Generated {len(preparation.get('optimizations', []))} optimizations")
    print(f"   • Identified {len(preparation.get('fixes', []))} potential fixes")
    print(f"   • Created {len(preparation.get('bundles', []))} workflow bundles")
    
    print("\n3. 📊 Ultimate metrics...")
    metrics = mechanic.get_ultimate_metrics()
    
    key_metrics = [
        'workflows_learned', 'optimizations_identified', 'prepared_workflows',
        'learning_confidence', 'workflows_optimized'
    ]
    
    for metric in key_metrics:
        if metric in metrics:
            print(f"   • {metric}: {metrics[metric]}")
    
    print("\n4. 🔍 Workflow optimization details...")
    for workflow in repo_info.get('workflows', []):
        optimizations = workflow.get('optimizations', [])
        if optimizations:
            print(f"   📝 {workflow.get('name', 'Unknown')}:")
            for opt in optimizations[:2]:  # Show first 2 optimizations
                print(f"     - {opt.get('type', 'unknown')}: {opt.get('description', 'No description')}")
    
    print("\n✅ Ultimate Cloud Mechanic Demo Complete!")
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(demo_ultimate_features())
        print(f"\n🎉 Demo {'succeeded' if result else 'failed'}!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)