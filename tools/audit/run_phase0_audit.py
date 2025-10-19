#!/usr/bin/env python3
"""
PHASE 0: Master Audit Script

Runs all Phase 0 audit scripts and generates a comprehensive report.
This is the entry point for the automated discovery and audit process.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import subprocess


def run_script(script_path: Path, description: str) -> bool:
    """Run a Python script and return success status"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            cwd=script_path.parent.parent
        )
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully")
            return True
        else:
            print(f"\n❌ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running {description}: {e}")
        return False


def generate_master_report(repo_root: Path) -> None:
    """Generate master report combining all audit results"""
    reports_dir = repo_root / "reports"
    
    report_files = {
        "training_systems": reports_dir / "training_systems_audit.json",
        "data_flow": reports_dir / "data_flow_analysis.json",
        "configuration": reports_dir / "configuration_validation.json",
        "safety_systems": reports_dir / "safety_systems_inventory.json"
    }
    
    master_report = {
        "generated_at": datetime.now().isoformat(),
        "phase": "PHASE 0: Discovery & Audit",
        "status": "COMPLETE",
        "reports": {}
    }
    
    all_success = True
    
    for report_name, report_path in report_files.items():
        if report_path.exists():
            try:
                with open(report_path, 'r') as f:
                    master_report["reports"][report_name] = json.load(f)
                print(f"   ✅ Loaded {report_name}")
            except Exception as e:
                print(f"   ⚠️  Error loading {report_name}: {e}")
                all_success = False
        else:
            print(f"   ❌ Missing {report_name}")
            all_success = False
            
    # Add summary analysis
    if all_success:
        master_report["summary"] = generate_summary(master_report["reports"])
        master_report["recommendations"] = generate_recommendations(master_report["reports"])
        
    # Save master report
    master_path = reports_dir / "phase0_master_audit.json"
    with open(master_path, 'w') as f:
        json.dump(master_report, f, indent=2)
        
    print(f"\n📊 Master report saved to: {master_path}")
    
    # Print executive summary
    print_executive_summary(master_report)


def generate_summary(reports: dict) -> dict:
    """Generate executive summary from all reports"""
    summary = {
        "audit_complete": True,
        "key_findings": []
    }
    
    # Training systems
    if "training_systems" in reports:
        ts = reports["training_systems"]["summary"]
        summary["key_findings"].append(
            f"Found {ts['total_training_methods']} training methods: "
            f"{ts['heavy_training']} heavy, {ts['medium_training']} medium, {ts['light_learning']} light"
        )
        summary["heavy_training_count"] = ts["heavy_training"]
        summary["light_learning_count"] = ts["light_learning"]
        
    # Configuration
    if "configuration" in reports:
        cfg = reports["configuration"]["summary"]
        summary["key_findings"].append(
            f"Configuration validation: {cfg['passed']} passed, "
            f"{cfg['warnings']} warnings, {cfg['failed']} failed"
        )
        summary["config_status"] = cfg["overall_status"]
        
    # Safety systems
    if "safety_systems" in reports:
        safety = reports["safety_systems"]["summary"]
        summary["key_findings"].append(
            f"Found {safety['total_safety_components']} safety-critical components "
            f"({safety['total_safety_code_lines']:,} lines) - all must stay in Live Mode"
        )
        summary["safety_components_count"] = safety["total_safety_components"]
        
    return summary


def generate_recommendations(reports: dict) -> list:
    """Generate actionable recommendations"""
    recommendations = []
    
    # Architecture recommendations
    recommendations.append({
        "priority": "HIGH",
        "area": "Architecture",
        "recommendation": "Split into Live Mode and Historical Mode",
        "rationale": "Heavy training methods are slowing down live trading decisions",
        "estimated_effort": "4-6 weeks for complete implementation"
    })
    
    # Training schedule recommendation
    if "training_systems" in reports:
        ts = reports["training_systems"]["summary"]
        if ts["heavy_training"] > 0:
            recommendations.append({
                "priority": "HIGH",
                "area": "Training Schedule",
                "recommendation": f"Move {ts['heavy_training']} heavy training methods to Sunday training window",
                "rationale": "Futures only have 1-hour daily maintenance, need 5+ hour window for training",
                "timeline": "Sunday 12 PM - 5:45 PM (before market open)"
            })
            
    # Live mode preservation
    recommendations.append({
        "priority": "CRITICAL",
        "area": "Safety",
        "recommendation": "Keep all safety systems in Live Mode only",
        "rationale": "Historical mode is offline - no broker connections, no real orders",
        "affected_systems": "TopStep enforcement, order execution, risk limits, position management"
    })
    
    # Configuration fixes
    if "configuration" in reports:
        cfg = reports["configuration"]["summary"]
        if cfg["failed"] > 0 or cfg["warnings"] > 0:
            recommendations.append({
                "priority": "MEDIUM",
                "area": "Configuration",
                "recommendation": f"Fix {cfg['failed']} failed checks and {cfg['warnings']} warnings",
                "rationale": "Ensure clean configuration before implementing split"
            })
            
    return recommendations


def print_executive_summary(report: dict) -> None:
    """Print executive summary to console"""
    print("\n" + "="*80)
    print("PHASE 0: EXECUTIVE SUMMARY")
    print("="*80)
    
    if "summary" in report:
        summary = report["summary"]
        print(f"\n✅ Audit Status: {summary.get('audit_complete', 'Unknown')}")
        print(f"\nKey Findings:")
        for finding in summary.get("key_findings", []):
            print(f"   • {finding}")
            
    if "recommendations" in report:
        print(f"\n📋 Top Recommendations:")
        for i, rec in enumerate(report["recommendations"][:5], 1):
            print(f"\n   {i}. [{rec['priority']}] {rec['area']}: {rec['recommendation']}")
            print(f"      Rationale: {rec['rationale']}")
            
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print("="*80)
    print("""
   1. Review all audit reports in the reports/ directory
   2. Discuss timeline and resource allocation with stakeholders  
   3. Begin Phase 1: Architecture Design (if proceeding with split)
   4. Create feature branch: git checkout -b feature/training-split
   
   All audit data is ready for architectural decision-making.
    """)


def main():
    """Main entry point"""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    
    print("="*80)
    print("PHASE 0: AUTOMATED DISCOVERY & AUDIT")
    print("="*80)
    print(f"\nRepository: {repo_root}")
    print(f"Audit Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis audit will:")
    print("   1. Discover all training systems and classify by complexity")
    print("   2. Trace data flow through the system")
    print("   3. Validate configuration for mode separation")
    print("   4. Inventory safety-critical systems")
    print("\nNo code will be modified - this is analysis only.\n")
    
    input("Press Enter to start the audit...")
    
    # Ensure reports directory exists
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Run all audit scripts
    audit_scripts = [
        (script_dir / "discover_training_systems.py", "Step 1: Training Systems Discovery"),
        (script_dir / "trace_data_flow.py", "Step 2: Data Flow Tracing"),
        (script_dir / "validate_configuration.py", "Step 3: Configuration Validation"),
        (script_dir / "inventory_safety_systems.py", "Step 4: Safety Systems Inventory"),
    ]
    
    results = []
    for script_path, description in audit_scripts:
        if script_path.exists():
            success = run_script(script_path, description)
            results.append(success)
        else:
            print(f"\n❌ Script not found: {script_path}")
            results.append(False)
            
    # Generate master report
    print(f"\n{'='*80}")
    print("Generating Master Report")
    print(f"{'='*80}\n")
    
    generate_master_report(repo_root)
    
    # Final status
    if all(results):
        print(f"\n✅ ALL AUDITS COMPLETED SUCCESSFULLY")
        return 0
    else:
        print(f"\n⚠️  SOME AUDITS FAILED - Review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
