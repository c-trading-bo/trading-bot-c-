#!/usr/bin/env python3
"""
PHASE 0 - STEP 1: Training Systems Discovery Script

This script automatically scans the codebase to identify all learning systems
and classifies them by complexity to determine which should run in live mode
vs historical training mode.

Classification:
- HEAVY: Multi-epoch training, gradient descent, backpropagation (→ Historical Mode)
- MEDIUM: Statistical updates, model retraining (→ Could fit in 15-min window)
- LIGHT: Simple statistics, immediate feedback (→ Live Mode)
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TrainingMethod:
    """Represents a training method found in the codebase"""
    file_path: str
    class_name: str
    method_name: str
    line_number: int
    complexity: str  # HEAVY, MEDIUM, LIGHT
    keywords_found: List[str]
    estimated_duration: str  # e.g., "Minutes to hours", "Seconds", "Milliseconds"
    code_snippet: str


class TrainingSystemsDiscovery:
    """Discovers and classifies training systems in the codebase"""
    
    # Keywords that indicate training operations
    HEAVY_KEYWORDS = [
        'gradient', 'backprop', 'epoch', 'batch', 'optimizer', 
        'loss.backward', 'train_step', 'fit(', 'compile(',
        'TrainAsync', 'MetaTrainAsync', 'neural network'
    ]
    
    MEDIUM_KEYWORDS = [
        'retrain', 'update_weights', 'statistical', 'refit',
        'calibrate', 'optimize', 'tune_parameters'
    ]
    
    LIGHT_KEYWORDS = [
        'online_learning', 'immediate_feedback', 'weight_update',
        'simple_average', 'running_mean', 'adaptive'
    ]
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.training_methods: List[TrainingMethod] = []
        
    def scan_codebase(self) -> None:
        """Scan all C# files in the src directory"""
        src_dir = self.repo_root / "src"
        
        print(f"🔍 Scanning {src_dir} for training systems...")
        
        cs_files = list(src_dir.rglob("*.cs"))
        print(f"   Found {len(cs_files)} C# files to analyze")
        
        for cs_file in cs_files:
            self._analyze_file(cs_file)
            
        print(f"✅ Discovered {len(self.training_methods)} training methods")
        
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze a single C# file for training methods"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            # Look for training-related methods
            for i, line in enumerate(lines, start=1):
                # Check for method signatures that suggest training
                if any(keyword.lower() in line.lower() for keyword in 
                      self.HEAVY_KEYWORDS + self.MEDIUM_KEYWORDS + self.LIGHT_KEYWORDS):
                    
                    # Extract context (class name, method name)
                    class_name = self._extract_class_name(lines, i)
                    method_name = self._extract_method_name(line)
                    
                    if method_name:
                        # Get code snippet (5 lines of context)
                        snippet = self._get_code_snippet(lines, i)
                        
                        # Classify complexity
                        complexity, keywords, duration = self._classify_complexity(snippet)
                        
                        training_method = TrainingMethod(
                            file_path=str(file_path.relative_to(self.repo_root)),
                            class_name=class_name,
                            method_name=method_name,
                            line_number=i,
                            complexity=complexity,
                            keywords_found=keywords,
                            estimated_duration=duration,
                            code_snippet=snippet
                        )
                        
                        self.training_methods.append(training_method)
                        
        except Exception as e:
            print(f"   ⚠️  Error analyzing {file_path}: {e}")
            
    def _extract_class_name(self, lines: List[str], current_line: int) -> str:
        """Extract the class name by looking backwards from current line"""
        for i in range(current_line - 1, max(0, current_line - 100), -1):
            line = lines[i].strip()
            if line.startswith('class ') or line.startswith('public class ') or \
               line.startswith('internal class ') or line.startswith('private class '):
                match = re.search(r'class\s+(\w+)', line)
                if match:
                    return match.group(1)
        return "Unknown"
        
    def _extract_method_name(self, line: str) -> str:
        """Extract method name from a line"""
        # Look for method signatures
        patterns = [
            r'(?:public|private|protected|internal)?\s+(?:async\s+)?(?:Task<?[\w]+>?|void|[\w]+)\s+(\w+)\s*\(',
            r'(?:public|private|protected|internal)?\s+(\w+)\s*\('
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                method_name = match.group(1)
                # Filter out common keywords that aren't methods
                if method_name not in ['if', 'for', 'while', 'switch', 'using', 'return']:
                    return method_name
        return ""
        
    def _get_code_snippet(self, lines: List[str], line_num: int, context_lines: int = 5) -> str:
        """Get code snippet with context"""
        start = max(0, line_num - context_lines)
        end = min(len(lines), line_num + context_lines)
        return '\n'.join(lines[start:end])
        
    def _classify_complexity(self, code_snippet: str) -> Tuple[str, List[str], str]:
        """Classify the complexity of training code"""
        snippet_lower = code_snippet.lower()
        found_keywords = []
        
        # Check for heavy keywords
        heavy_count = sum(1 for kw in self.HEAVY_KEYWORDS if kw.lower() in snippet_lower)
        if heavy_count > 0:
            found_keywords.extend([kw for kw in self.HEAVY_KEYWORDS if kw.lower() in snippet_lower])
            
        # Check for medium keywords
        medium_count = sum(1 for kw in self.MEDIUM_KEYWORDS if kw.lower() in snippet_lower)
        if medium_count > 0:
            found_keywords.extend([kw for kw in self.MEDIUM_KEYWORDS if kw.lower() in snippet_lower])
            
        # Check for light keywords
        light_count = sum(1 for kw in self.LIGHT_KEYWORDS if kw.lower() in snippet_lower)
        if light_count > 0:
            found_keywords.extend([kw for kw in self.LIGHT_KEYWORDS if kw.lower() in snippet_lower])
        
        # Classify based on keyword counts and patterns
        if heavy_count >= 2 or 'trainasync' in snippet_lower or 'metatrainasync' in snippet_lower:
            return "HEAVY", found_keywords, "Minutes to hours (needs Sunday training)"
        elif 'epoch' in snippet_lower or 'gradient' in snippet_lower or 'backprop' in snippet_lower:
            return "HEAVY", found_keywords, "Minutes to hours (needs Sunday training)"
        elif medium_count >= 2 or heavy_count == 1:
            return "MEDIUM", found_keywords, "Seconds to minutes (could fit in 15-min window)"
        elif light_count > 0 or 'online' in snippet_lower:
            return "LIGHT", found_keywords, "Milliseconds (stays in live mode)"
        else:
            # Default to MEDIUM if we can't clearly classify
            return "MEDIUM", found_keywords, "Uncertain - needs manual review"
            
    def generate_report(self, output_path: str) -> None:
        """Generate JSON report of findings"""
        # Group by complexity
        heavy = [m for m in self.training_methods if m.complexity == "HEAVY"]
        medium = [m for m in self.training_methods if m.complexity == "MEDIUM"]
        light = [m for m in self.training_methods if m.complexity == "LIGHT"]
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "repository_root": str(self.repo_root),
            "summary": {
                "total_training_methods": len(self.training_methods),
                "heavy_training": len(heavy),
                "medium_training": len(medium),
                "light_learning": len(light)
            },
            "recommendations": {
                "heavy_training": "Must run in Historical Mode (Sunday 12 PM - 5:45 PM)",
                "medium_training": "Could run in daily maintenance window (15 min)",
                "light_learning": "Keep in Live Mode for immediate adaptation"
            },
            "heavy_training_methods": [asdict(m) for m in heavy],
            "medium_training_methods": [asdict(m) for m in medium],
            "light_learning_methods": [asdict(m) for m in light]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📊 REPORT SUMMARY:")
        print(f"   Total training methods found: {len(self.training_methods)}")
        print(f"   HEAVY (→ Historical Mode): {len(heavy)}")
        print(f"   MEDIUM (→ Maybe daily window): {len(medium)}")
        print(f"   LIGHT (→ Live Mode): {len(light)}")
        print(f"\n💾 Full report saved to: {output_file}")
        
        # Print top 5 heavy training methods
        if heavy:
            print(f"\n🔴 TOP HEAVY TRAINING METHODS (need Sunday training):")
            for i, method in enumerate(heavy[:5], 1):
                print(f"   {i}. {method.class_name}.{method.method_name}")
                print(f"      File: {method.file_path}:{method.line_number}")
                print(f"      Keywords: {', '.join(method.keywords_found[:3])}")


def main():
    """Main entry point"""
    import sys
    
    # Get repository root (assume script is in tools/audit/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    
    print("=" * 80)
    print("PHASE 0 - STEP 1: Training Systems Discovery")
    print("=" * 80)
    
    discovery = TrainingSystemsDiscovery(repo_root)
    discovery.scan_codebase()
    
    output_path = repo_root / "reports" / "training_systems_audit.json"
    discovery.generate_report(output_path)
    
    print("\n✅ Discovery complete!")
    print(f"   Review the report at: {output_path}")
    print(f"   This data will inform the split between live and historical modes.\n")


if __name__ == "__main__":
    main()
