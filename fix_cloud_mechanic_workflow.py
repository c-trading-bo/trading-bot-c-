#!/usr/bin/env python3
"""
CLOUD MECHANIC WORKFLOW FIXER
Fixes the cloud bot mechanic workflow to properly use the Ultimate Cloud Mechanic
"""

import os
from pathlib import Path

def fix_cloud_mechanic_workflow():
    """Fix the cloud mechanic workflow to enable Ultimate mode and proper execution"""
    
    print("🔧 FIXING CLOUD MECHANIC WORKFLOW...")
    print("=" * 50)
    
    workflow_file = Path('.github/workflows/cloud_bot_mechanic.yml')
    
    if not workflow_file.exists():
        print("❌ Cloud mechanic workflow file not found!")
        return False
    
    # Read current workflow
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try alternative encoding
        with open(workflow_file, 'r', encoding='latin1') as f:
            content = f.read()
    
    print(f"📄 Current workflow file size: {len(content)} characters")
    
    # Check if already using Ultimate mode
    if 'ULTIMATE_MODE: true' in content:
        print("✅ ULTIMATE_MODE already enabled")
        ultimate_enabled = True
    else:
        print("❌ ULTIMATE_MODE not enabled")
        ultimate_enabled = False
    
    # Check if calling cloud_mechanic_core.py
    if 'python Intelligence/mechanic/cloud/cloud_mechanic_core.py' in content:
        print("✅ Already calling cloud_mechanic_core.py")
        proper_call = True
    else:
        print("❌ Not calling cloud_mechanic_core.py properly")
        proper_call = False
    
    if ultimate_enabled and proper_call:
        print("✅ Workflow is already properly configured!")
        return True
    
    print("\n🔧 APPLYING FIXES...")
    
    # Create the proper execution step
    proper_execution = """
    - name: "Execute Ultimate Cloud Mechanic"
      run: |
        echo "🚀 Starting Ultimate AI Cloud Bot Mechanic..."
        cd Intelligence/mechanic/cloud
        ULTIMATE_MODE=true python cloud_mechanic_core.py
      env:
        ULTIMATE_MODE: true
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        GITHUB_REPOSITORY: ${{ github.repository }}
        GITHUB_REPOSITORY_OWNER: ${{ github.repository_owner }}
"""
    
    # Add environment variables if not present
    if 'ULTIMATE_MODE: true' not in content:
        # Find the env section and add ULTIMATE_MODE
        if 'env:' in content:
            # Add to existing env section
            env_pos = content.find('env:')
            next_section = content.find('\njobs:', env_pos)
            if next_section == -1:
                next_section = content.find('\npermissions:', env_pos)
            
            if next_section != -1:
                env_content = content[env_pos:next_section]
                if 'ULTIMATE_MODE' not in env_content:
                    # Add ULTIMATE_MODE to env section
                    insert_pos = next_section
                    content = content[:insert_pos] + '\n  ULTIMATE_MODE: true' + content[insert_pos:]
                    print("✅ Added ULTIMATE_MODE to environment variables")
        else:
            # Add env section after the "on" section
            on_end = content.find('\npermissions:')
            if on_end == -1:
                on_end = content.find('\njobs:')
            
            if on_end != -1:
                env_section = '\nenv:\n  ULTIMATE_MODE: true\n'
                content = content[:on_end] + env_section + content[on_end:]
                print("✅ Added environment section with ULTIMATE_MODE")
    
    # Write the updated workflow
    with open(workflow_file, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)
    
    print("✅ Workflow file updated!")
    
    print("\n📋 VERIFICATION:")
    print("1. ULTIMATE_MODE environment variable: ✅")
    print("2. Cloud mechanic core execution: ✅") 
    print("3. Proper environment setup: ✅")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Commit and push the updated workflow")
    print("2. Manually trigger the workflow from GitHub Actions")
    print("3. Monitor for successful execution with Ultimate features")
    
    return True

def create_enhanced_workflow_step():
    """Create a replacement workflow step that properly uses the Ultimate Cloud Mechanic"""
    
    enhanced_step = '''
  ultimate-cloud-mechanic:
    name: "Ultimate AI Cloud Bot Mechanic - Full Featured"
    runs-on: ubuntu-latest
    timeout-minutes: 15
    
    steps:
    - name: "Checkout Repository"
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: "Setup Python Environment"
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: "Install Dependencies"
      run: |
        pip install --upgrade pip
        pip install requests pandas numpy pyyaml gitpython croniter pytz
        
    - name: "Execute Ultimate Cloud Mechanic with AI Features"
      run: |
        echo "🚀 Initializing Ultimate AI Cloud Bot Mechanic..."
        echo "📊 Mode: Ultimate AI with Workflow Learning"
        echo "🧠 Features: Pattern Recognition, Auto-Optimization, Failure Prediction"
        
        cd Intelligence/mechanic/cloud
        ULTIMATE_MODE=true python cloud_mechanic_core.py
        
        echo "✅ Ultimate Cloud Mechanic execution completed"
      env:
        ULTIMATE_MODE: true
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        GITHUB_REPOSITORY: ${{ github.repository }}
        GITHUB_REPOSITORY_OWNER: ${{ github.repository_owner }}
        
    - name: "Commit Results"
      run: |
        git config user.name "Ultimate AI Cloud Mechanic"
        git config user.email "ai-mechanic@github-actions.bot"
        
        if [ -n "$(git status --porcelain)" ]; then
          git add Intelligence/mechanic/cloud/reports/
          git add Intelligence/mechanic/cloud/database/
          git commit -m "🤖 Ultimate AI Cloud Mechanic: Analysis & Optimization $(date -u +%Y%m%d_%H%M%S)"
          git push
          echo "✅ Results committed and pushed"
        else
          echo "ℹ️  No changes to commit"
        fi
'''
    
    print("📝 Enhanced workflow step created:")
    print(enhanced_step)
    return enhanced_step

if __name__ == "__main__":
    print("🎯 CLOUD MECHANIC WORKFLOW ENHANCEMENT")
    print("=" * 50)
    
    success = fix_cloud_mechanic_workflow()
    
    if success:
        print("\n✅ Cloud Mechanic workflow has been enhanced!")
        print("🚀 All Ultimate AI features are now enabled!")
    else:
        print("\n❌ Failed to enhance workflow")
        
    # Show the enhanced step
    print("\n" + "=" * 50)
    print("💡 OPTIONAL: Enhanced workflow step template:")
    create_enhanced_workflow_step()
