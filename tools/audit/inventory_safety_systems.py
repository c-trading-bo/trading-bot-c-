#!/usr/bin/env python3
"""
PHASE 0 - STEP 4: Safety Systems Inventory

Identifies all safety-critical code that must remain in live mode:
- TopStep enforcement
- Broker connection code
- Order execution logic
- Position management
- Risk limits

Marks these with [LiveModeOnly] conceptually for architectural planning.
"""

import json
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SafetyComponent:
    """Represents a safety-critical component"""
    component_name: str
    file_path: str
    component_type: str  # enforcement, connection, execution, risk, position
    line_count: int
    critical_methods: List[str]
    must_stay_in_live_mode: bool
    reason: str


class SafetySystemInventory:
    """Inventories safety-critical systems"""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.components: List[SafetyComponent] = []
        
    def scan_safety_systems(self) -> None:
        """Scan for safety-critical systems"""
        print("🔍 Scanning for safety-critical systems...")
        
        # Define safety-critical areas
        safety_areas = {
            "TopStep Enforcement": {
                "patterns": ["topstep", "compliance", "rule", "violation"],
                "dirs": ["src/TopstepAuthAgent", "src/Safety"]
            },
            "Broker Connection": {
                "patterns": ["connect", "authenticate", "websocket", "adapter"],
                "dirs": ["src/adapters", "src/UnifiedOrchestrator/TopstepX"]
            },
            "Order Execution": {
                "patterns": ["placeorder", "execution", "fill", "order"],
                "dirs": ["src/BotCore/Execution", "src/UnifiedOrchestrator/Services"]
            },
            "Risk Management": {
                "patterns": ["risk", "limit", "drawdown", "breach"],
                "dirs": ["src/Safety", "src/BotCore"]
            },
            "Position Management": {
                "patterns": ["position", "breakeven", "trailing", "stop"],
                "dirs": ["src/BotCore"]
            }
        }
        
        for area_name, config in safety_areas.items():
            self._scan_area(area_name, config)
            
        print(f"✅ Found {len(self.components)} safety-critical components")
        
    def _scan_area(self, area_name: str, config: Dict) -> None:
        """Scan a specific safety area"""
        patterns = config["patterns"]
        dirs = config["dirs"]
        
        for dir_path in dirs:
            full_dir = self.repo_root / dir_path
            if not full_dir.exists():
                continue
                
            cs_files = list(full_dir.rglob("*.cs"))
            for cs_file in cs_files:
                self._analyze_safety_file(cs_file, area_name, patterns)
                
    def _analyze_safety_file(self, file_path: Path, area: str, patterns: List[str]) -> None:
        """Analyze a file for safety-critical code"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            content_lower = content.lower()
            
            # Check if file contains safety patterns
            matches = sum(1 for pattern in patterns if pattern.lower() in content_lower)
            
            if matches == 0:
                return
                
            # Extract class name
            class_name = "Unknown"
            for line in lines:
                if 'class ' in line:
                    parts = line.split('class ')
                    if len(parts) > 1:
                        class_name = parts[1].split()[0].split(':')[0].split('<')[0]
                        break
                        
            # Find critical methods
            critical_methods = []
            for i, line in enumerate(lines):
                line_lower = line.lower()
                if any(pattern in line_lower for pattern in patterns):
                    method = self._extract_method(line)
                    if method and method not in critical_methods:
                        critical_methods.append(method)
                        
            # Determine component type
            component_type = self._categorize_component(area, file_path)
            
            # All safety components must stay in live mode
            reason = self._get_safety_reason(component_type)
            
            component = SafetyComponent(
                component_name=class_name,
                file_path=str(file_path.relative_to(self.repo_root)),
                component_type=component_type,
                line_count=len(lines),
                critical_methods=critical_methods[:10],  # Limit to first 10
                must_stay_in_live_mode=True,
                reason=reason
            )
            
            self.components.append(component)
            
        except Exception as e:
            print(f"   ⚠️  Error analyzing {file_path}: {e}")
            
    def _extract_method(self, line: str) -> str:
        """Extract method name from line"""
        import re
        patterns = [
            r'public\s+\w+\s+(\w+)\s*\(',
            r'private\s+\w+\s+(\w+)\s*\(',
            r'internal\s+\w+\s+(\w+)\s*\(',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return match.group(1)
        return ""
        
    def _categorize_component(self, area: str, file_path: Path) -> str:
        """Categorize the component type"""
        area_lower = area.lower()
        path_lower = str(file_path).lower()
        
        if "topstep" in area_lower or "compliance" in area_lower:
            return "enforcement"
        elif "connection" in area_lower or "adapter" in path_lower:
            return "connection"
        elif "execution" in area_lower or "order" in path_lower:
            return "execution"
        elif "risk" in area_lower:
            return "risk"
        elif "position" in area_lower:
            return "position"
        else:
            return "safety"
            
    def _get_safety_reason(self, component_type: str) -> str:
        """Get reason why component must stay in live mode"""
        reasons = {
            "enforcement": "TopStep compliance rules must be enforced in real-time",
            "connection": "Live broker connection required for order execution",
            "execution": "Order execution must happen during live trading only",
            "risk": "Risk limits must be enforced in real-time",
            "position": "Position management requires live market data",
            "safety": "Safety-critical code must run in live mode"
        }
        return reasons.get(component_type, "Safety-critical component")
        
    def generate_report(self, output_path: str) -> None:
        """Generate safety inventory report"""
        # Group by type
        by_type = {}
        for comp in self.components:
            if comp.component_type not in by_type:
                by_type[comp.component_type] = []
            by_type[comp.component_type].append(comp)
            
        total_lines = sum(c.line_count for c in self.components)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "repository_root": str(self.repo_root),
            "summary": {
                "total_safety_components": len(self.components),
                "total_safety_code_lines": total_lines,
                "components_by_type": {k: len(v) for k, v in by_type.items()},
                "all_must_stay_in_live_mode": all(c.must_stay_in_live_mode for c in self.components)
            },
            "safety_components": [asdict(c) for c in self.components],
            "architectural_note": (
                "All safety-critical components MUST remain in Live Mode. "
                "Historical training mode operates completely offline with no "
                "broker connections, no order execution, and no safety enforcement "
                "needed (it only processes historical data)."
            )
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📊 SAFETY SYSTEMS INVENTORY:")
        print(f"   Total components: {len(self.components)}")
        print(f"   Total lines of safety code: {total_lines:,}")
        print(f"\n   By type:")
        for comp_type, comps in sorted(by_type.items()):
            print(f"      {comp_type}: {len(comps)} components")
            
        print(f"\n   ⚠️  ALL safety components must stay in Live Mode")
        print(f"   ✅ Historical mode will NOT include any safety/execution code")
        print(f"\n💾 Report saved to: {output_file}")


def main():
    """Main entry point"""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    
    print("=" * 80)
    print("PHASE 0 - STEP 4: Safety Systems Inventory")
    print("=" * 80)
    
    inventory = SafetySystemInventory(repo_root)
    inventory.scan_safety_systems()
    
    output_path = repo_root / "reports" / "safety_systems_inventory.json"
    inventory.generate_report(output_path)
    
    print("\n✅ Safety inventory complete!\n")


if __name__ == "__main__":
    main()
