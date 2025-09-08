#!/usr/bin/env python3
"""
REPORT GENERATOR - Generates comprehensive reports
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class ReportGenerator:
    def __init__(self):
        self.cloud_path = Path('Intelligence/mechanic/cloud')
        
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of all mechanic activities"""
        print("\n📊 GENERATING COMPREHENSIVE REPORT...")
        
        # Load all data
        workflows_data = self.load_json(self.cloud_path / 'database' / 'workflows.json', {})
        analysis_data = self.load_json(self.cloud_path / 'database' / 'detailed_analysis.json', {})
        optimizations_data = self.load_json(self.cloud_path / 'database' / 'optimizations.json', {})
        repairs_data = self.load_json(self.cloud_path / 'database' / 'repairs.json', {})
        
        # Generate report
        report = {
            'meta': {
                'generated_at': datetime.utcnow().isoformat(),
                'generator_version': '3.0-CLOUD-ULTIMATE',
                'report_type': 'comprehensive'
            },
            'summary': self.generate_summary(workflows_data, analysis_data, optimizations_data, repairs_data),
            'workflow_health': self.analyze_workflow_health(workflows_data),
            'performance_metrics': self.calculate_performance_metrics(workflows_data, optimizations_data),
            'recent_activities': self.compile_recent_activities(optimizations_data, repairs_data),
            'recommendations': self.generate_recommendations(workflows_data, analysis_data)
        }
        
        # Save report
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = self.cloud_path / 'reports' / f'comprehensive_report_{timestamp}.json'
        self.save_json(report_path, report)
        
        # Save as latest
        latest_path = self.cloud_path / 'reports' / 'latest_comprehensive_report.json'
        self.save_json(latest_path, report)
        
        # Generate markdown report
        markdown_report = self.generate_markdown_report(report)
        markdown_path = self.cloud_path / 'reports' / f'report_{timestamp}.md'
        self.save_text(markdown_path, markdown_report)
        
        self.print_report_summary(report)
        return report
    
    def generate_summary(self, workflows, analysis, optimizations, repairs) -> Dict:
        """Generate high-level summary"""
        total_workflows = len([k for k, v in workflows.items() if isinstance(v, dict) and k != 'last_analysis'])
        
        # Get health data
        last_analysis = workflows.get('last_analysis', {})
        healthy_workflows = last_analysis.get('healthy_workflows', 0)
        total_minutes = last_analysis.get('total_monthly_minutes', 0)
        
        # Get activity data
        optimizations_count = len(optimizations.get('optimizations_applied', []))
        repairs_count = len(repairs.get('repairs_successful', []))
        minutes_saved = optimizations.get('minutes_saved_estimate', 0)
        
        return {
            'total_workflows': total_workflows,
            'healthy_workflows': healthy_workflows,
            'broken_workflows': total_workflows - healthy_workflows,
            'health_percentage': round((healthy_workflows / max(total_workflows, 1)) * 100, 1),
            'monthly_minutes_usage': total_minutes,
            'minutes_under_limit': 20000 - total_minutes,
            'recent_optimizations': optimizations_count,
            'recent_repairs': repairs_count,
            'estimated_minutes_saved': minutes_saved
        }
    
    def analyze_workflow_health(self, workflows_data) -> Dict:
        """Analyze overall workflow health"""
        health_data = {
            'healthy': [],
            'needs_attention': [],
            'broken': [],
            'high_usage': []
        }
        
        for wf_name, wf_data in workflows_data.items():
            if wf_name == 'last_analysis' or not isinstance(wf_data, dict):
                continue
            
            issues = wf_data.get('issues', [])
            monthly_minutes = wf_data.get('runs_per_month', 0) * wf_data.get('estimated_minutes', 5)
            
            if not issues:
                health_data['healthy'].append(wf_name)
            elif len(issues) == 1:
                health_data['needs_attention'].append({'name': wf_name, 'issues': issues})
            else:
                health_data['broken'].append({'name': wf_name, 'issues': issues})
            
            if monthly_minutes > 1000:
                health_data['high_usage'].append({
                    'name': wf_name,
                    'monthly_minutes': monthly_minutes
                })
        
        return health_data
    
    def calculate_performance_metrics(self, workflows_data, optimizations_data) -> Dict:
        """Calculate performance metrics"""
        total_jobs = 0
        total_steps = 0
        cached_workflows = 0
        
        for wf_name, wf_data in workflows_data.items():
            if wf_name == 'last_analysis' or not isinstance(wf_data, dict):
                continue
            
            jobs = wf_data.get('jobs', {})
            total_jobs += len(jobs)
            
            for job_data in jobs.values():
                if isinstance(job_data, dict):
                    total_steps += job_data.get('steps', 0)
                    if job_data.get('has_cache', False):
                        cached_workflows += 1
                        break  # Count workflow only once
        
        return {
            'total_jobs': total_jobs,
            'total_steps': total_steps,
            'avg_jobs_per_workflow': round(total_jobs / max(len(workflows_data) - 1, 1), 1),
            'avg_steps_per_job': round(total_steps / max(total_jobs, 1), 1),
            'cached_workflows': cached_workflows,
            'optimization_opportunities': len(optimizations_data.get('optimizations_applied', []))
        }
    
    def compile_recent_activities(self, optimizations_data, repairs_data) -> Dict:
        """Compile recent activities"""
        activities = {
            'optimizations': optimizations_data.get('optimizations_applied', [])[-10:],  # Last 10
            'repairs': repairs_data.get('repairs_made', [])[-10:],  # Last 10
            'files_modified': list(set(
                optimizations_data.get('files_modified', []) + 
                repairs_data.get('files_modified', [])
            )),
            'files_created': repairs_data.get('files_created', [])
        }
        
        return activities
    
    def generate_recommendations(self, workflows_data, analysis_data) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Get analysis data
        last_analysis = workflows_data.get('last_analysis', {})
        total_minutes = last_analysis.get('total_monthly_minutes', 0)
        broken_workflows = last_analysis.get('broken_workflows', [])
        
        # High priority recommendations
        if total_minutes > 18000:
            recommendations.append("🚨 HIGH PRIORITY: Reduce workflow frequency - approaching minute limit")
        
        if len(broken_workflows) > 0:
            recommendations.append(f"🔧 FIX REQUIRED: {len(broken_workflows)} broken workflows need attention")
        
        # Optimization recommendations
        if total_minutes > 15000:
            recommendations.extend([
                "⚡ Implement caching across all workflows",
                "⚡ Use matrix strategies for similar jobs",
                "⚡ Add conditional execution to reduce unnecessary runs"
            ])
        
        # Monitoring recommendations
        workflows_without_timeouts = 0
        for wf_data in workflows_data.values():
            if isinstance(wf_data, dict) and wf_data.get('jobs'):
                for job_data in wf_data['jobs'].values():
                    if isinstance(job_data, dict) and job_data.get('timeout') == 360:
                        workflows_without_timeouts += 1
                        break
        
        if workflows_without_timeouts > 0:
            recommendations.append(f"⏱️ Set explicit timeouts for {workflows_without_timeouts} workflows")
        
        # General recommendations
        total_workflows = len([k for k, v in workflows_data.items() if isinstance(v, dict) and k != 'last_analysis'])
        if total_workflows > 50:
            recommendations.append("📊 Consider workflow consolidation for large repository")
        
        return recommendations[:10]  # Limit to top 10
    
    def generate_markdown_report(self, report: Dict) -> str:
        """Generate a markdown version of the report"""
        summary = report['summary']
        health = report['workflow_health']
        metrics = report['performance_metrics']
        activities = report['recent_activities']
        recommendations = report['recommendations']
        
        md = f"""# 🌩️ Cloud Bot Mechanic Report
        
Generated: {report['meta']['generated_at']}

## 📊 Summary

- **Total Workflows**: {summary['total_workflows']}
- **Health Score**: {summary['health_percentage']}% ({summary['healthy_workflows']}/{summary['total_workflows']} healthy)
- **Monthly Minutes**: {summary['monthly_minutes_usage']:,}/20,000 ({summary['minutes_under_limit']:,} remaining)
- **Recent Activity**: {summary['recent_optimizations']} optimizations, {summary['recent_repairs']} repairs

## 🏥 Workflow Health

### ✅ Healthy Workflows ({len(health['healthy'])})
{chr(10).join(f"- {wf}" for wf in health['healthy'][:10])}
{f"... and {len(health['healthy']) - 10} more" if len(health['healthy']) > 10 else ""}

### ⚠️ Needs Attention ({len(health['needs_attention'])})
{chr(10).join(f"- **{item['name']}**: {', '.join(item['issues'])}" for item in health['needs_attention'][:5])}

### ❌ Broken Workflows ({len(health['broken'])})
{chr(10).join(f"- **{item['name']}**: {', '.join(item['issues'])}" for item in health['broken'][:5])}

## 📈 Performance Metrics

- **Total Jobs**: {metrics['total_jobs']}
- **Total Steps**: {metrics['total_steps']}
- **Average Jobs per Workflow**: {metrics['avg_jobs_per_workflow']}
- **Cached Workflows**: {metrics['cached_workflows']}

## 🔧 Recent Activities

### Optimizations Applied
{chr(10).join(f"- {opt}" for opt in activities['optimizations'][-5:])}

### Repairs Made
{chr(10).join(f"- {repair}" for repair in activities['repairs'][-5:])}

## 💡 Recommendations

{chr(10).join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations))}

---
*Generated by Cloud Bot Mechanic v{report['meta']['generator_version']}*
"""
        return md
    
    def print_report_summary(self, report: Dict):
        """Print a summary of the generated report"""
        summary = report['summary']
        
        print("📊 REPORT GENERATED")
        print(f"   Total workflows: {summary['total_workflows']}")
        print(f"   Health score: {summary['health_percentage']}%")
        print(f"   Monthly minutes: {summary['monthly_minutes_usage']:,}/20,000")
        print(f"   Recommendations: {len(report['recommendations'])}")
        
        # Show top recommendation
        if report['recommendations']:
            print(f"   Top recommendation: {report['recommendations'][0]}")
    
    def load_json(self, path: Path, default):
        """Load JSON with default fallback"""
        try:
            if path.exists():
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return default
    
    def save_json(self, path: Path, data: Dict):
        """Save JSON data"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save {path}: {e}")
    
    def save_text(self, path: Path, content: str):
        """Save text content"""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"⚠️ Failed to save {path}: {e}")

if __name__ == "__main__":
    generator = ReportGenerator()
    generator.generate_comprehensive_report()
