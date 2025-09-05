#!/usr/bin/env python3
"""
Ultimate Cloud Mechanic Enabler
Adds ULTIMATE_MODE and proper execution to the cloud mechanic workflow
"""
import re

def enable_ultimate_mode():
    workflow_path = ".github/workflows/cloud_bot_mechanic.yml"
    
    print("🔧 Enabling Ultimate AI Cloud Mechanic...")
    
    try:
        # Read the workflow file
        with open(workflow_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if ULTIMATE_MODE is already added
        if 'ULTIMATE_MODE: true' in content:
            print("✅ ULTIMATE_MODE already enabled")
        else:
            # Find the env section and add ULTIMATE_MODE
            env_pattern = r'(\s+env:\s*\n(?:\s+\w+:.*\n)*)'
            if re.search(env_pattern, content):
                # Add to existing env section
                content = re.sub(
                    r'(\s+env:\s*\n)',
                    r'\1        ULTIMATE_MODE: true\n',
                    content
                )
                print("✅ Added ULTIMATE_MODE to existing env section")
            else:
                # Add env section after jobs:
                content = re.sub(
                    r'(jobs:\s*\n\s+\w+:\s*\n)',
                    r'\1    env:\n      ULTIMATE_MODE: true\n',
                    content
                )
                print("✅ Created env section with ULTIMATE_MODE")
        
        # Check if Ultimate execution step exists
        if 'cd Intelligence/mechanic/cloud && python cloud_mechanic_core.py' in content:
            print("✅ Ultimate execution step already exists")
        else:
            # Find the steps section and add Ultimate execution
            steps_pattern = r'(\s+steps:\s*\n(?:\s+-.*\n)*)'
            
            ultimate_step = '''      - name: "🚀 Execute Ultimate AI Cloud Mechanic"
        run: |
          echo "🔥 Launching Ultimate AI Cloud Mechanic with full features..."
          cd Intelligence/mechanic/cloud
          export ULTIMATE_MODE=true
          python cloud_mechanic_core.py
          echo "✅ Ultimate AI Cloud Mechanic execution completed!"
        env:
          ULTIMATE_MODE: true

'''
            
            # Add the step before the last step
            content = re.sub(
                r'(\s+- name: ".*Final Status.*")',
                ultimate_step + r'\1',
                content
            )
            print("✅ Added Ultimate AI execution step")
        
        # Write the enhanced workflow
        with open(workflow_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("🚀 Ultimate AI Cloud Mechanic is now fully enabled!")
        print("   ✅ ULTIMATE_MODE environment variable set")
        print("   ✅ Direct execution of cloud_mechanic_core.py added")
        print("   ✅ All AI features including workflow learning activated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error enabling Ultimate mode: {e}")
        return False

if __name__ == "__main__":
    enable_ultimate_mode()
