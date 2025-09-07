#!/usr/bin/env python3
"""
COMPLETE TRADING BOT SYSTEM ANALYSIS
===================================

This script performs a comprehensive analysis of the entire trading bot system,
examining every department, service, configuration, and integration point.
"""

import os
import json
import subprocess
import glob
from pathlib import Path
from datetime import datetime

class CompleteBotAnalyzer:
    def __init__(self, bot_root):
        self.bot_root = Path(bot_root)
        self.analysis_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def analyze_complete_system(self):
        """Perform complete system analysis"""
        print("🤖 COMPLETE TRADING BOT SYSTEM ANALYSIS")
        print("=" * 50)
        print(f"📅 Analysis Date: {datetime.now()}")
        print(f"📁 Bot Root: {self.bot_root}")
        print("\n")
        
        # 1. Project Structure Analysis
        self.analyze_project_structure()
        
        # 2. Source Code Analysis
        self.analyze_source_code()
        
        # 3. Configuration Analysis
        self.analyze_configurations()
        
        # 4. Dependencies Analysis
        self.analyze_dependencies()
        
        # 5. Build System Analysis
        self.analyze_build_system()
        
        # 6. Environment Analysis
        self.analyze_environment()
        
        # 7. GitHub Integration Analysis
        self.analyze_github_integration()
        
        # 8. ML/AI Components Analysis
        self.analyze_ml_components()
        
        # 9. Security Analysis
        self.analyze_security()
        
        # 10. Performance Analysis
        self.analyze_performance()
        
        # Generate comprehensive report
        self.generate_complete_report()
        
    def analyze_project_structure(self):
        """Analyze complete project structure"""
        print("📁 ANALYZING PROJECT STRUCTURE...")
        
        structure = {
            "total_files": 0,
            "directories": [],
            "file_types": {},
            "src_projects": [],
            "config_files": [],
            "scripts": [],
            "documentation": []
        }
        
        # Walk through entire project
        for root, dirs, files in os.walk(self.bot_root):
            rel_root = os.path.relpath(root, self.bot_root)
            structure["directories"].append(rel_root)
            
            for file in files:
                structure["total_files"] += 1
                ext = Path(file).suffix.lower()
                structure["file_types"][ext] = structure["file_types"].get(ext, 0) + 1
                
                # Categorize files
                if file.endswith(('.csproj', '.sln')):
                    structure["src_projects"].append(os.path.join(rel_root, file))
                elif file.startswith('.env') or file.endswith(('.json', '.yml', '.yaml')):
                    structure["config_files"].append(os.path.join(rel_root, file))
                elif file.endswith(('.py', '.ps1', '.sh', '.bat')):
                    structure["scripts"].append(os.path.join(rel_root, file))
                elif file.endswith(('.md', '.txt', '.rst')):
                    structure["documentation"].append(os.path.join(rel_root, file))
        
        self.analysis_results["project_structure"] = structure
        
        print(f"   📊 Total Files: {structure['total_files']}")
        print(f"   📁 Directories: {len(structure['directories'])}")
        print(f"   🔧 C# Projects: {len(structure['src_projects'])}")
        print(f"   ⚙️ Config Files: {len(structure['config_files'])}")
        print(f"   📝 Scripts: {len(structure['scripts'])}")
        print(f"   📚 Documentation: {len(structure['documentation'])}")
        print()
        
    def analyze_source_code(self):
        """Analyze all source code components"""
        print("💻 ANALYZING SOURCE CODE...")
        
        src_analysis = {
            "csharp_projects": {},
            "python_scripts": {},
            "javascript_files": {},
            "total_lines": 0,
            "complexity_metrics": {}
        }
        
        # Analyze C# projects
        for csproj in glob.glob(str(self.bot_root / "**/*.csproj"), recursive=True):
            project_name = Path(csproj).stem
            project_dir = Path(csproj).parent
            
            cs_files = list(project_dir.glob("**/*.cs"))
            total_lines = 0
            
            for cs_file in cs_files:
                try:
                    with open(cs_file, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                except:
                    pass
            
            src_analysis["csharp_projects"][project_name] = {
                "path": str(project_dir),
                "cs_files": len(cs_files),
                "total_lines": total_lines
            }
            src_analysis["total_lines"] += total_lines
        
        # Analyze Python scripts
        for py_file in glob.glob(str(self.bot_root / "**/*.py"), recursive=True):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    src_analysis["python_scripts"][Path(py_file).name] = {
                        "path": py_file,
                        "lines": lines
                    }
                    src_analysis["total_lines"] += lines
            except:
                pass
        
        self.analysis_results["source_code"] = src_analysis
        
        print(f"   🔷 C# Projects: {len(src_analysis['csharp_projects'])}")
        print(f"   🐍 Python Scripts: {len(src_analysis['python_scripts'])}")
        print(f"   📏 Total Lines of Code: {src_analysis['total_lines']:,}")
        print()
        
    def analyze_configurations(self):
        """Analyze all configuration files"""
        print("⚙️ ANALYZING CONFIGURATIONS...")
        
        config_analysis = {
            "env_files": {},
            "json_configs": {},
            "csproj_configs": {},
            "other_configs": {}
        }
        
        # Analyze .env files
        for env_file in glob.glob(str(self.bot_root / ".env*")):
            try:
                with open(env_file, 'r') as f:
                    content = f.read()
                    vars_count = len([line for line in content.split('\n') 
                                    if '=' in line and not line.strip().startswith('#')])
                config_analysis["env_files"][Path(env_file).name] = {
                    "path": env_file,
                    "variables": vars_count,
                    "size": len(content)
                }
            except:
                pass
        
        # Analyze JSON configs
        for json_file in glob.glob(str(self.bot_root / "**/*.json"), recursive=True):
            if "node_modules" not in json_file and ".git" not in json_file:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        config_analysis["json_configs"][Path(json_file).name] = {
                            "path": json_file,
                            "keys": len(data) if isinstance(data, dict) else 0
                        }
                except:
                    pass
        
        self.analysis_results["configurations"] = config_analysis
        
        print(f"   🌍 Environment Files: {len(config_analysis['env_files'])}")
        print(f"   📋 JSON Configs: {len(config_analysis['json_configs'])}")
        print()
        
    def analyze_dependencies(self):
        """Analyze all dependencies and packages"""
        print("📦 ANALYZING DEPENDENCIES...")
        
        deps_analysis = {
            "nuget_packages": {},
            "python_packages": {},
            "npm_packages": {},
            "total_dependencies": 0
        }
        
        # Analyze NuGet packages (C#)
        for csproj in glob.glob(str(self.bot_root / "**/*.csproj"), recursive=True):
            try:
                with open(csproj, 'r') as f:
                    content = f.read()
                    # Count PackageReference entries
                    package_refs = content.count('<PackageReference')
                    deps_analysis["nuget_packages"][Path(csproj).stem] = package_refs
                    deps_analysis["total_dependencies"] += package_refs
            except:
                pass
        
        # Analyze Python packages
        for req_file in ["requirements.txt", "requirements_ml.txt", "requirements_bulletproof.txt"]:
            req_path = self.bot_root / req_file
            if req_path.exists():
                try:
                    with open(req_path, 'r') as f:
                        packages = [line.strip() for line in f.readlines() 
                                  if line.strip() and not line.startswith('#')]
                        deps_analysis["python_packages"][req_file] = len(packages)
                        deps_analysis["total_dependencies"] += len(packages)
                except:
                    pass
        
        self.analysis_results["dependencies"] = deps_analysis
        
        print(f"   📦 Total Dependencies: {deps_analysis['total_dependencies']}")
        print(f"   🔷 NuGet Projects: {len(deps_analysis['nuget_packages'])}")
        print(f"   🐍 Python Requirements: {len(deps_analysis['python_packages'])}")
        print()
        
    def analyze_build_system(self):
        """Analyze build system and compilation"""
        print("🔨 ANALYZING BUILD SYSTEM...")
        
        build_analysis = {
            "solution_files": [],
            "project_files": [],
            "build_scripts": [],
            "can_build": False,
            "build_errors": []
        }
        
        # Find solution files
        for sln in glob.glob(str(self.bot_root / "*.sln")):
            build_analysis["solution_files"].append(sln)
        
        # Find project files
        for csproj in glob.glob(str(self.bot_root / "**/*.csproj"), recursive=True):
            build_analysis["project_files"].append(csproj)
        
        # Test build capability
        if build_analysis["solution_files"] or build_analysis["project_files"]:
            try:
                # Try to restore packages
                result = subprocess.run(
                    ["dotnet", "restore"], 
                    cwd=self.bot_root, 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                if result.returncode == 0:
                    build_analysis["can_build"] = True
                else:
                    build_analysis["build_errors"].append(result.stderr)
            except Exception as e:
                build_analysis["build_errors"].append(str(e))
        
        self.analysis_results["build_system"] = build_analysis
        
        print(f"   📁 Solution Files: {len(build_analysis['solution_files'])}")
        print(f"   🔧 Project Files: {len(build_analysis['project_files'])}")
        print(f"   ✅ Can Build: {build_analysis['can_build']}")
        print()
        
    def analyze_environment(self):
        """Analyze environment and runtime"""
        print("🌍 ANALYZING ENVIRONMENT...")
        
        env_analysis = {
            "dotnet_version": None,
            "python_version": None,
            "git_status": None,
            "environment_vars": {},
            "disk_usage": None
        }
        
        # Check .NET version
        try:
            result = subprocess.run(["dotnet", "--version"], capture_output=True, text=True)
            env_analysis["dotnet_version"] = result.stdout.strip()
        except:
            pass
        
        # Check Python version
        try:
            result = subprocess.run(["python", "--version"], capture_output=True, text=True)
            env_analysis["python_version"] = result.stdout.strip()
        except:
            pass
        
        # Check Git status
        try:
            result = subprocess.run(["git", "status", "--porcelain"], 
                                  cwd=self.bot_root, capture_output=True, text=True)
            env_analysis["git_status"] = "clean" if not result.stdout.strip() else "modified"
        except:
            pass
        
        # Check disk usage
        try:
            total_size = sum(f.stat().st_size for f in Path(self.bot_root).rglob('*') if f.is_file())
            env_analysis["disk_usage"] = f"{total_size / (1024*1024):.1f} MB"
        except:
            pass
        
        self.analysis_results["environment"] = env_analysis
        
        print(f"   🔷 .NET Version: {env_analysis['dotnet_version']}")
        print(f"   🐍 Python Version: {env_analysis['python_version']}")
        print(f"   📂 Git Status: {env_analysis['git_status']}")
        print(f"   💾 Disk Usage: {env_analysis['disk_usage']}")
        print()
        
    def analyze_github_integration(self):
        """Analyze GitHub integration and workflows"""
        print("🐙 ANALYZING GITHUB INTEGRATION...")
        
        github_analysis = {
            "workflows": [],
            "github_configs": [],
            "has_git": False,
            "branch_info": None
        }
        
        # Check .github directory
        github_dir = self.bot_root / ".github"
        if github_dir.exists():
            # Find workflows
            workflows_dir = github_dir / "workflows"
            if workflows_dir.exists():
                for workflow in workflows_dir.glob("*.yml"):
                    github_analysis["workflows"].append(workflow.name)
        
        # Check Git repository
        git_dir = self.bot_root / ".git"
        if git_dir.exists():
            github_analysis["has_git"] = True
            try:
                result = subprocess.run(["git", "branch", "--show-current"], 
                                      cwd=self.bot_root, capture_output=True, text=True)
                github_analysis["branch_info"] = result.stdout.strip()
            except:
                pass
        
        self.analysis_results["github_integration"] = github_analysis
        
        print(f"   🔄 GitHub Workflows: {len(github_analysis['workflows'])}")
        print(f"   📂 Git Repository: {github_analysis['has_git']}")
        print(f"   🌿 Current Branch: {github_analysis['branch_info']}")
        print()
        
    def analyze_ml_components(self):
        """Analyze ML/AI components"""
        print("🧠 ANALYZING ML/AI COMPONENTS...")
        
        ml_analysis = {
            "ml_directories": [],
            "model_files": [],
            "training_scripts": [],
            "data_files": [],
            "rl_components": []
        }
        
        # Find ML directories
        ml_dirs = ["ml", "models", "data", "Intelligence", "RL"]
        for ml_dir in ml_dirs:
            dir_path = self.bot_root / ml_dir
            if dir_path.exists():
                ml_analysis["ml_directories"].append(ml_dir)
        
        # Find model files
        model_extensions = ['.pkl', '.joblib', '.h5', '.onnx', '.pt', '.pth']
        for ext in model_extensions:
            for model_file in self.bot_root.rglob(f"*{ext}"):
                ml_analysis["model_files"].append(str(model_file))
        
        # Find training scripts
        for py_file in self.bot_root.rglob("*.py"):
            if any(keyword in py_file.name.lower() for keyword in ['train', 'model', 'ml', 'rl']):
                ml_analysis["training_scripts"].append(str(py_file))
        
        self.analysis_results["ml_components"] = ml_analysis
        
        print(f"   📁 ML Directories: {len(ml_analysis['ml_directories'])}")
        print(f"   🤖 Model Files: {len(ml_analysis['model_files'])}")
        print(f"   📝 Training Scripts: {len(ml_analysis['training_scripts'])}")
        print()
        
    def analyze_security(self):
        """Analyze security configurations"""
        print("🔒 ANALYZING SECURITY...")
        
        security_analysis = {
            "credential_files": [],
            "api_keys_found": False,
            "secure_storage": False,
            "gitignore_exists": False,
            "sensitive_patterns": []
        }
        
        # Check for credential files
        cred_patterns = [".env*", "*credentials*", "*secrets*", "*keys*"]
        for pattern in cred_patterns:
            for file in self.bot_root.rglob(pattern):
                if file.is_file():
                    security_analysis["credential_files"].append(str(file))
        
        # Check .gitignore
        gitignore_path = self.bot_root / ".gitignore"
        security_analysis["gitignore_exists"] = gitignore_path.exists()
        
        # Check for API keys in code (basic scan)
        sensitive_patterns = ["api_key", "secret", "password", "token"]
        for cs_file in self.bot_root.rglob("*.cs"):
            try:
                with open(cs_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for pattern in sensitive_patterns:
                        if pattern in content:
                            security_analysis["sensitive_patterns"].append(f"{cs_file.name}:{pattern}")
                            break
            except:
                pass
        
        self.analysis_results["security"] = security_analysis
        
        print(f"   🔑 Credential Files: {len(security_analysis['credential_files'])}")
        print(f"   🚫 GitIgnore Exists: {security_analysis['gitignore_exists']}")
        print(f"   ⚠️ Sensitive Patterns: {len(security_analysis['sensitive_patterns'])}")
        print()
        
    def analyze_performance(self):
        """Analyze performance characteristics"""
        print("⚡ ANALYZING PERFORMANCE...")
        
        perf_analysis = {
            "large_files": [],
            "code_complexity": {},
            "memory_usage_patterns": [],
            "optimization_opportunities": []
        }
        
        # Find large files
        for file in self.bot_root.rglob("*"):
            if file.is_file():
                try:
                    size = file.stat().st_size
                    if size > 1024 * 1024:  # > 1MB
                        perf_analysis["large_files"].append({
                            "file": str(file),
                            "size_mb": round(size / (1024*1024), 2)
                        })
                except:
                    pass
        
        self.analysis_results["performance"] = perf_analysis
        
        print(f"   📁 Large Files (>1MB): {len(perf_analysis['large_files'])}")
        print()
        
    def generate_complete_report(self):
        """Generate comprehensive analysis report"""
        print("📊 GENERATING COMPLETE REPORT...")
        
        report_file = self.bot_root / f"COMPLETE_BOT_ANALYSIS_{self.timestamp}.json"
        
        # Save detailed analysis
        with open(report_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Generate summary report
        summary_file = self.bot_root / f"BOT_ANALYSIS_SUMMARY_{self.timestamp}.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# COMPLETE TRADING BOT ANALYSIS REPORT\n")
            f.write(f"**Generated:** {datetime.now()}\n\n")
            
            # Project Overview
            f.write("## 📊 PROJECT OVERVIEW\n")
            f.write(f"- **Total Files:** {self.analysis_results['project_structure']['total_files']:,}\n")
            f.write(f"- **Lines of Code:** {self.analysis_results['source_code']['total_lines']:,}\n")
            f.write(f"- **C# Projects:** {len(self.analysis_results['source_code']['csharp_projects'])}\n")
            f.write(f"- **Dependencies:** {self.analysis_results['dependencies']['total_dependencies']}\n")
            f.write(f"- **Disk Usage:** {self.analysis_results['environment']['disk_usage']}\n\n")
            
            # System Health
            f.write("## 🏥 SYSTEM HEALTH\n")
            f.write(f"- **Can Build:** {'✅' if self.analysis_results['build_system']['can_build'] else '❌'}\n")
            f.write(f"- **Git Status:** {self.analysis_results['environment']['git_status']}\n")
            f.write(f"- **Security:** {'✅' if self.analysis_results['security']['gitignore_exists'] else '⚠️'}\n\n")
            
            # Components Summary
            f.write("## 🏗️ COMPONENTS SUMMARY\n")
            for project, info in self.analysis_results['source_code']['csharp_projects'].items():
                f.write(f"- **{project}:** {info['cs_files']} files, {info['total_lines']:,} lines\n")
        
        print(f"   📄 Detailed Report: {report_file}")
        print(f"   📋 Summary Report: {summary_file}")
        print()
        
        return report_file, summary_file

def main():
    """Main analysis function"""
    bot_root = r"C:\Users\kevin\Downloads\C# ai bot"
    
    if not os.path.exists(bot_root):
        print(f"❌ Bot directory not found: {bot_root}")
        return
    
    analyzer = CompleteBotAnalyzer(bot_root)
    analyzer.analyze_complete_system()
    
    print("🎉 COMPLETE ANALYSIS FINISHED!")
    print("Check the generated reports for detailed findings.")

if __name__ == "__main__":
    main()
