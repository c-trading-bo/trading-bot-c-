#!/usr/bin/env python3
"""
PHASE 0 - STEP 2: Data Flow Tracing Script

This script instruments the codebase to trace the experience lifecycle:
- When experience is created
- When it's saved to storage
- When it's loaded by trainer
- When it's processed for learning

Generates a flow diagram showing data movement.
"""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DataFlowNode:
    """Represents a node in the data flow"""
    node_id: str
    node_type: str  # creation, storage, loading, processing
    component: str
    file_path: str
    line_number: int
    description: str


@dataclass
class DataFlowEdge:
    """Represents a connection in the data flow"""
    from_node: str
    to_node: str
    data_type: str  # experience, bar, feature, model
    description: str


class DataFlowTracer:
    """Traces data flow through the trading system"""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.nodes: List[DataFlowNode] = []
        self.edges: List[DataFlowEdge] = []
        
    def analyze_experience_lifecycle(self) -> None:
        """Analyze how experiences flow through the system"""
        print("🔍 Analyzing experience lifecycle...")
        
        # Key files to analyze
        key_files = {
            "UnifiedTradingBrain": "src/BotCore/Brain/UnifiedTradingBrain.cs",
            "EnhancedBacktestLearning": "src/UnifiedOrchestrator/Services/EnhancedBacktestLearningService.cs",
            "CVaRPPO": "src/RLAgent/CVaRPPO.cs",
            "OnlineLearning": "src/IntelligenceStack/OnlineLearningSystem.cs"
        }
        
        for component, file_path in key_files.items():
            full_path = self.repo_root / file_path
            if full_path.exists():
                self._analyze_component(component, full_path)
                
        # Define known data flows based on architecture documentation
        self._define_known_flows()
        
        print(f"✅ Found {len(self.nodes)} data flow nodes")
        print(f"✅ Found {len(self.edges)} data flow edges")
        
    def _analyze_component(self, component: str, file_path: Path) -> None:
        """Analyze a component for data flow patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines, start=1):
                line_lower = line.lower()
                
                # Look for experience creation
                if 'new experience' in line_lower or 'addexperience' in line_lower:
                    self.nodes.append(DataFlowNode(
                        node_id=f"{component}_create_{i}",
                        node_type="creation",
                        component=component,
                        file_path=str(file_path.relative_to(self.repo_root)),
                        line_number=i,
                        description=f"Experience created in {component}"
                    ))
                    
                # Look for storage operations
                if 'save' in line_lower or 'write' in line_lower or 'log' in line_lower:
                    if 'experience' in line_lower or 'buffer' in line_lower:
                        self.nodes.append(DataFlowNode(
                            node_id=f"{component}_save_{i}",
                            node_type="storage",
                            component=component,
                            file_path=str(file_path.relative_to(self.repo_root)),
                            line_number=i,
                            description=f"Experience saved in {component}"
                        ))
                        
                # Look for loading operations
                if 'load' in line_lower or 'read' in line_lower or 'get' in line_lower:
                    if 'experience' in line_lower or 'historical' in line_lower:
                        self.nodes.append(DataFlowNode(
                            node_id=f"{component}_load_{i}",
                            node_type="loading",
                            component=component,
                            file_path=str(file_path.relative_to(self.repo_root)),
                            line_number=i,
                            description=f"Data loaded in {component}"
                        ))
                        
                # Look for training operations
                if 'train' in line_lower or 'learn' in line_lower or 'update' in line_lower:
                    if 'async' in line_lower or 'from' in line_lower:
                        self.nodes.append(DataFlowNode(
                            node_id=f"{component}_process_{i}",
                            node_type="processing",
                            component=component,
                            file_path=str(file_path.relative_to(self.repo_root)),
                            line_number=i,
                            description=f"Learning in {component}"
                        ))
                        
        except Exception as e:
            print(f"   ⚠️  Error analyzing {component}: {e}")
            
    def _define_known_flows(self) -> None:
        """Define known data flows based on architecture"""
        # Live trading flow
        self.edges.append(DataFlowEdge(
            from_node="live_market_data",
            to_node="UnifiedTradingBrain",
            data_type="bar",
            description="Market bars fed to decision brain"
        ))
        
        self.edges.append(DataFlowEdge(
            from_node="UnifiedTradingBrain",
            to_node="experience_buffer",
            data_type="experience",
            description="Decisions logged as experiences"
        ))
        
        # Historical training flow
        self.edges.append(DataFlowEdge(
            from_node="historical_seed",
            to_node="EnhancedBacktestLearning",
            data_type="bar",
            description="90-day historical data loaded"
        ))
        
        self.edges.append(DataFlowEdge(
            from_node="EnhancedBacktestLearning",
            to_node="UnifiedTradingBrain",
            data_type="bar",
            description="Historical bars replayed through brain"
        ))
        
        self.edges.append(DataFlowEdge(
            from_node="experience_buffer",
            to_node="CVaRPPO",
            data_type="experience",
            description="Experiences used for training"
        ))
        
        self.edges.append(DataFlowEdge(
            from_node="CVaRPPO",
            to_node="model_registry",
            data_type="model",
            description="Trained models saved"
        ))
        
    def generate_report(self, output_path: str) -> None:
        """Generate data flow report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "repository_root": str(self.repo_root),
            "summary": {
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "node_types": {
                    "creation": len([n for n in self.nodes if n.node_type == "creation"]),
                    "storage": len([n for n in self.nodes if n.node_type == "storage"]),
                    "loading": len([n for n in self.nodes if n.node_type == "loading"]),
                    "processing": len([n for n in self.nodes if n.node_type == "processing"])
                }
            },
            "data_flow_nodes": [asdict(n) for n in self.nodes],
            "data_flow_edges": [asdict(e) for e in self.edges],
            "flow_diagram_mermaid": self._generate_mermaid_diagram()
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📊 DATA FLOW ANALYSIS:")
        print(f"   Nodes found: {len(self.nodes)}")
        print(f"   - Creation points: {report['summary']['node_types']['creation']}")
        print(f"   - Storage points: {report['summary']['node_types']['storage']}")
        print(f"   - Loading points: {report['summary']['node_types']['loading']}")
        print(f"   - Processing points: {report['summary']['node_types']['processing']}")
        print(f"\n💾 Report saved to: {output_file}")
        
    def _generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid diagram of data flow"""
        lines = ["graph TD"]
        
        # Add edges
        for edge in self.edges:
            from_label = edge.from_node.replace("_", "")
            to_label = edge.to_node.replace("_", "")
            lines.append(f"    {from_label}[\"{edge.from_node}\"] --> |{edge.data_type}| {to_label}[\"{edge.to_node}\"]")
            
        return "\n".join(lines)


def main():
    """Main entry point"""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    
    print("=" * 80)
    print("PHASE 0 - STEP 2: Data Flow Tracing")
    print("=" * 80)
    
    tracer = DataFlowTracer(repo_root)
    tracer.analyze_experience_lifecycle()
    
    output_path = repo_root / "reports" / "data_flow_analysis.json"
    tracer.generate_report(output_path)
    
    print("\n✅ Data flow analysis complete!")
    print(f"   Review the report at: {output_path}\n")


if __name__ == "__main__":
    main()
