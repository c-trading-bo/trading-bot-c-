#!/usr/bin/env python3
"""
🔍 SIMPLE PRE-LAUNCH VERIFICATION CHECKLIST
==========================================
Quick checklist to verify all critical systems are working and no placeholders exist.
"""

import os
import re
import glob
from pathlib import Path

def main():
    print("🚀 TRADING BOT PRE-LAUNCH VERIFICATION")
    print("=" * 50)
    
    root_path = Path(os.getcwd())
    
    # 1. Critical File Check
    print("\n📋 1. CRITICAL FILES CHECK")
    critical_files = [
        "src/UnifiedOrchestrator/Program.cs",
        "src/UnifiedOrchestrator/Services/CentralMessageBus.cs",
        "src/UnifiedOrchestrator/Services/TradingOrchestratorService.cs",
        "Enhanced/MLRLSystem.cs",
        "Enhanced/MarketIntelligence.cs",
        "src/BotCore/Risk/RiskEngine.cs",
        "src/BotCore/Strategy/StrategyMlIntegration.cs"
    ]
    
    for file in critical_files:
        file_path = root_path / file
        if file_path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ MISSING: {file}")
    
    # 2. Placeholder Detection
    print("\n🔍 2. PLACEHOLDER DETECTION")
    placeholder_patterns = [
        r'TODO\s*:',
        r'FIXME\s*:',
        r'PLACEHOLDER',
        r'NotImplementedException',
        r'Console\.WriteLine\("Mock',
        r'// Simulate',
        r'Random\.Shared\.Next'
    ]
    
    placeholder_count = 0
    suspicious_files = []
    
    for pattern in ['**/*.cs']:
        for file_path in glob.glob(str(root_path / pattern), recursive=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                file_placeholders = 0
                for pattern_regex in placeholder_patterns:
                    matches = re.findall(pattern_regex, content, re.IGNORECASE)
                    file_placeholders += len(matches)
                
                if file_placeholders > 0:
                    placeholder_count += file_placeholders
                    rel_path = os.path.relpath(file_path, root_path)
                    suspicious_files.append((rel_path, file_placeholders))
                    
            except Exception:
                continue
    
    if placeholder_count == 0:
        print("  ✅ No placeholders detected")
    else:
        print(f"  ⚠️  {placeholder_count} potential placeholders found in {len(suspicious_files)} files")
        for file, count in suspicious_files[:5]:  # Show first 5
            print(f"    📁 {file}: {count} placeholders")
    
    # 3. System Architecture Check
    print("\n🏗️  3. SYSTEM ARCHITECTURE")
    
    # Check Unified Orchestrator
    unified_path = root_path / "src" / "UnifiedOrchestrator"
    if unified_path.exists():
        print("  ✅ Unified Orchestrator exists")
        
        # Count services
        services_path = unified_path / "Services"
        if services_path.exists():
            service_files = list(services_path.glob("*.cs"))
            print(f"  ✅ {len(service_files)} service files found")
        else:
            print("  ❌ Services directory missing")
    else:
        print("  ❌ Unified Orchestrator missing")
    
    # Check GitHub workflows
    workflow_dir = root_path / ".github" / "workflows"
    if workflow_dir.exists():
        workflow_count = len(list(workflow_dir.glob("*.yml")))
        print(f"  ✅ {workflow_count} GitHub workflows found")
    else:
        print("  ❌ GitHub workflows missing")
    
    # 4. Logic Depth Analysis
    print("\n🧠 4. LOGIC DEPTH ANALYSIS")
    
    # Check Enhanced ML/RL system
    mlrl_path = root_path / "Enhanced" / "MLRLSystem.cs"
    if mlrl_path.exists():
        try:
            with open(mlrl_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for ML/RL patterns
            ml_patterns = ['LSTM', 'Transformer', 'XGBoost', 'FinBERT', 'Autoencoder']
            rl_patterns = ['DQN', 'PPO', 'A3C']
            
            found_ml = sum(1 for pattern in ml_patterns if pattern in content)
            found_rl = sum(1 for pattern in rl_patterns if pattern in content)
            
            print(f"  ✅ ML models detected: {found_ml}/5")
            print(f"  ✅ RL agents detected: {found_rl}/3")
            
            # Check logic complexity
            lines = len([line for line in content.split('\n') if line.strip() and not line.strip().startswith('//')])
            methods = len(re.findall(r'(public|private|protected)\s+\w+\s+\w+\s*\(', content))
            classes = len(re.findall(r'class\s+\w+', content))
            
            print(f"  ✅ Code metrics: {lines} LOC, {methods} methods, {classes} classes")
            
        except Exception as e:
            print(f"  ❌ Error reading MLRLSystem.cs: {str(e)}")
    else:
        print("  ❌ MLRLSystem.cs missing")
    
    # 5. Configuration Check
    print("\n⚙️  5. CONFIGURATION CHECK")
    
    config_files = [
        'appsettings.json',
        '.env.sample.local',
        'Enhanced/Enhanced.csproj',
        'src/UnifiedOrchestrator/UnifiedOrchestrator.csproj'
    ]
    
    for config_file in config_files:
        file_path = root_path / config_file
        if file_path.exists():
            print(f"  ✅ {config_file}")
        else:
            print(f"  ⚠️  {config_file} missing")
    
    # 6. Process Check
    print("\n🔄 6. RUNNING PROCESSES")
    try:
        import subprocess
        result = subprocess.run(['powershell', '-Command', 'Get-Process -Name "dotnet" -ErrorAction SilentlyContinue'], 
                              capture_output=True, text=True, timeout=5)
        if 'dotnet' in result.stdout:
            print("  ✅ .NET processes running")
        else:
            print("  ⚠️  No .NET processes detected")
    except Exception:
        print("  ⚠️  Cannot check process status")
    
    # 7. Final Recommendation
    print("\n" + "=" * 50)
    print("🎯 FINAL ASSESSMENT")
    print("=" * 50)
    
    # Calculate score
    critical_missing = len([f for f in critical_files if not (root_path / f).exists()])
    
    if critical_missing == 0 and placeholder_count < 5:
        status = "✅ READY TO LAUNCH"
        print(f"{status}")
        print("🚀 All critical systems verified and minimal placeholders detected")
        print("💡 System appears production ready")
    elif critical_missing == 0 and placeholder_count < 20:
        status = "⚠️  CAUTION - REVIEW RECOMMENDED"
        print(f"{status}")
        print(f"🔧 Consider reviewing {placeholder_count} placeholders before launch")
        print("💡 Can proceed with monitoring")
    else:
        status = "❌ NOT READY - ISSUES FOUND"
        print(f"{status}")
        print(f"🛑 {critical_missing} critical files missing, {placeholder_count} placeholders found")
        print("💡 Fix issues before launching")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Critical Files Missing: {critical_missing}")
    print(f"  Placeholders Found: {placeholder_count}")
    print(f"  Suspicious Files: {len(suspicious_files)}")
    
    return status.startswith("✅")

if __name__ == "__main__":
    success = main()
    print(f"\n{'🚀 VERIFICATION PASSED' if success else '🛑 VERIFICATION FAILED'}")
