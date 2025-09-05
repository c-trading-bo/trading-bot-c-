#!/usr/bin/env python3
"""
Add Ultimate AI Cloud Mechanic to the latest workflow
"""

def add_ultimate_ai_step():
    workflow_path = ".github/workflows/cloud_bot_mechanic.yml"
    
    print("🚀 Adding Ultimate AI Cloud Mechanic step...")
    
    try:
        # Read the current workflow
        with open(workflow_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📄 Current file size: {len(content)} characters")
        
        # Define the Ultimate AI step
        ultimate_step = '''
      - name: "🚀 Execute Ultimate AI Cloud Mechanic"
        run: |
          echo "🔥 Launching Ultimate AI Cloud Mechanic with full features..."
          if [ -f "Intelligence/mechanic/cloud/cloud_mechanic_core.py" ]; then
            cd Intelligence/mechanic/cloud
            export ULTIMATE_MODE=true
            python cloud_mechanic_core.py
            echo "✅ Ultimate AI Cloud Mechanic execution completed!"
          else
            echo "⚠️ Ultimate AI Cloud Mechanic not found, using built-in emergency mode"
          fi
        env:
          ULTIMATE_MODE: true
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
'''
        
        # Find where to insert (after Install Dependencies)
        insert_marker = 'pip install openai anthropic'
        
        if insert_marker in content:
            # Insert the Ultimate AI step after dependencies
            parts = content.split(insert_marker)
            if len(parts) == 2:
                new_content = parts[0] + insert_marker + ultimate_step + parts[1]
                
                # Write the enhanced workflow
                with open(workflow_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                print("✅ Ultimate AI Cloud Mechanic step added successfully!")
                print("   🔥 Will execute cloud_mechanic_core.py with ULTIMATE_MODE=true")
                print("   🧠 All AI features including workflow learning activated")
                return True
            else:
                print("❌ Multiple insertion points found")
                return False
        else:
            print("❌ Dependencies section not found")
            return False
        
    except Exception as e:
        print(f"❌ Error adding Ultimate AI step: {e}")
        return False

if __name__ == "__main__":
    add_ultimate_ai_step()
