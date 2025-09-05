#!/usr/bin/env python3
"""
Cloud Mechanic Workflow Restorer
Fixes the empty workflow file bug
"""

def restore_cloud_mechanic():
    import shutil
    import os
    
    backup_path = r"C:\Users\kevin\Downloads\C# ai bot\.github\workflows_backup\cloud_bot_mechanic.yml"
    target_path = r"C:\Users\kevin\Downloads\C# ai bot\.github\workflows\cloud_bot_mechanic.yml"
    
    print("🔧 Restoring Cloud Mechanic workflow...")
    
    try:
        # Check backup exists
        if not os.path.exists(backup_path):
            print("❌ Backup file not found!")
            return False
            
        # Get backup size
        backup_size = os.path.getsize(backup_path)
        print(f"📁 Backup file size: {backup_size} bytes")
        
        # Read backup content
        with open(backup_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📄 Content length: {len(content)} characters")
        print(f"🔍 First line: {content.split(chr(10))[0] if content else 'EMPTY'}")
        
        # Write to target
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Verify
        target_size = os.path.getsize(target_path)
        print(f"✅ Target file size: {target_size} bytes")
        
        if target_size == backup_size:
            print("🚀 Cloud Mechanic workflow successfully restored!")
            return True
        else:
            print("⚠️ Size mismatch - may be an issue")
            return False
            
    except Exception as e:
        print(f"❌ Error restoring workflow: {e}")
        return False

if __name__ == "__main__":
    restore_cloud_mechanic()
