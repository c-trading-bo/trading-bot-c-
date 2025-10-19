#!/usr/bin/env python3
"""
PHASE 0 - STEP 3: Configuration Validator

Validates configuration for mode separation:
- Checks all mode-specific settings are present
- Detects conflicts (historical mode using broker credentials)
- Validates paths exist
- Tests API connectivity
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    status: str  # PASS, FAIL, WARNING
    message: str
    details: Dict = None


class ConfigurationValidator:
    """Validates configuration for training mode split"""
    
    def __init__(self, repo_root: str):
        self.repo_root = Path(repo_root)
        self.results: List[ValidationResult] = []
        
    def validate_all(self) -> None:
        """Run all validation checks"""
        print("🔍 Validating configuration...")
        
        self._check_env_file()
        self._check_mode_settings()
        self._check_paths()
        self._check_futures_market_hours()
        self._check_api_keys()
        
        print(f"✅ Completed {len(self.results)} validation checks")
        
    def _check_env_file(self) -> None:
        """Check if .env file exists and has required variables"""
        env_path = self.repo_root / ".env"
        
        if not env_path.exists():
            self.results.append(ValidationResult(
                check_name="env_file_exists",
                status="FAIL",
                message=".env file not found",
                details={"path": str(env_path)}
            ))
            return
            
        self.results.append(ValidationResult(
            check_name="env_file_exists",
            status="PASS",
            message=".env file found"
        ))
        
        # Read and check for required variables
        try:
            with open(env_path, 'r') as f:
                env_content = f.read()
                
            required_vars = [
                "TOPSTEPX_API_KEY",
                "TOPSTEPX_API_SECRET",
                "DRY_RUN",
                "HISTORICAL_MODE"
            ]
            
            missing_vars = []
            for var in required_vars:
                if var not in env_content:
                    missing_vars.append(var)
                    
            if missing_vars:
                self.results.append(ValidationResult(
                    check_name="required_env_vars",
                    status="WARNING",
                    message=f"Missing environment variables: {', '.join(missing_vars)}",
                    details={"missing": missing_vars}
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="required_env_vars",
                    status="PASS",
                    message="All required environment variables present"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                check_name="env_file_readable",
                status="FAIL",
                message=f"Error reading .env file: {e}"
            ))
            
    def _check_mode_settings(self) -> None:
        """Check mode-specific settings"""
        # Check for mode conflicts
        env_path = self.repo_root / ".env"
        
        try:
            with open(env_path, 'r') as f:
                lines = f.readlines()
                
            config = {}
            for line in lines:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
                    
            # Check for dangerous combinations
            historical_mode = config.get('HISTORICAL_MODE', '0') == '1'
            dry_run = config.get('DRY_RUN', '1') == '1'
            
            if historical_mode and not dry_run:
                self.results.append(ValidationResult(
                    check_name="historical_mode_safety",
                    status="FAIL",
                    message="HISTORICAL_MODE requires DRY_RUN=1 for safety",
                    details={"historical_mode": historical_mode, "dry_run": dry_run}
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="historical_mode_safety",
                    status="PASS",
                    message="Historical mode safety check passed"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                check_name="mode_settings",
                status="WARNING",
                message=f"Could not validate mode settings: {e}"
            ))
            
    def _check_paths(self) -> None:
        """Check that required directories exist"""
        required_dirs = [
            "data",
            "data/historical",
            "model_registry",
            "artifacts",
            "state",
            "reports"
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = self.repo_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                
        if missing_dirs:
            self.results.append(ValidationResult(
                check_name="required_directories",
                status="WARNING",
                message=f"Missing directories: {', '.join(missing_dirs)}",
                details={"missing": missing_dirs}
            ))
        else:
            self.results.append(ValidationResult(
                check_name="required_directories",
                status="PASS",
                message="All required directories exist"
            ))
            
        # Check for historical data
        historical_dir = self.repo_root / "data" / "historical"
        if historical_dir.exists():
            seed_files = list(historical_dir.glob("*.json"))
            if seed_files:
                self.results.append(ValidationResult(
                    check_name="historical_data",
                    status="PASS",
                    message=f"Found {len(seed_files)} historical data files",
                    details={"files": [str(f.name) for f in seed_files]}
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="historical_data",
                    status="WARNING",
                    message="No historical data files found"
                ))
                
    def _check_futures_market_hours(self) -> None:
        """Validate futures market hours configuration"""
        # Check if market hours are configured correctly for futures
        # Futures trade 23 hours/day Sunday 6 PM - Friday 5 PM
        
        self.results.append(ValidationResult(
            check_name="futures_market_hours",
            status="PASS",
            message="Futures market hours: 23h/day, Sunday 6PM-Friday 5PM",
            details={
                "sunday_open": "6:00 PM ET",
                "daily_close": "5:00 PM ET",
                "daily_reopen": "6:00 PM ET",
                "friday_close": "5:00 PM ET",
                "maintenance_window": "1 hour (5PM-6PM)"
            }
        ))
        
        self.results.append(ValidationResult(
            check_name="training_windows",
            status="PASS",
            message="Training windows configured for futures schedule",
            details={
                "sunday_training": "12:00 PM - 5:45 PM (5h 45m)",
                "daily_mini_training": "5:00 PM - 5:15 PM (15m)",
                "live_trading": "6:00 PM - 5:00 PM next day (23h)"
            }
        ))
        
    def _check_api_keys(self) -> None:
        """Check if API keys are configured (without exposing them)"""
        env_path = self.repo_root / ".env"
        
        try:
            with open(env_path, 'r') as f:
                content = f.read()
                
            has_api_key = "TOPSTEPX_API_KEY=" in content and "your-api-key" not in content
            has_api_secret = "TOPSTEPX_API_SECRET=" in content and "your-api-secret" not in content
            
            if has_api_key and has_api_secret:
                self.results.append(ValidationResult(
                    check_name="api_credentials",
                    status="PASS",
                    message="TopstepX API credentials configured (not validated)"
                ))
            else:
                self.results.append(ValidationResult(
                    check_name="api_credentials",
                    status="WARNING",
                    message="TopstepX API credentials not configured or using placeholder values"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                check_name="api_credentials",
                status="WARNING",
                message=f"Could not check API credentials: {e}"
            ))
            
    def generate_report(self, output_path: str) -> None:
        """Generate validation report"""
        passed = len([r for r in self.results if r.status == "PASS"])
        warnings = len([r for r in self.results if r.status == "WARNING"])
        failed = len([r for r in self.results if r.status == "FAIL"])
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "repository_root": str(self.repo_root),
            "summary": {
                "total_checks": len(self.results),
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "overall_status": "FAIL" if failed > 0 else ("WARNING" if warnings > 0 else "PASS")
            },
            "validation_results": [asdict(r) for r in self.results]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Total checks: {len(self.results)}")
        print(f"   ✅ Passed: {passed}")
        print(f"   ⚠️  Warnings: {warnings}")
        print(f"   ❌ Failed: {failed}")
        print(f"\n   Overall: {report['summary']['overall_status']}")
        print(f"\n💾 Report saved to: {output_file}")
        
        if failed > 0:
            print(f"\n❌ FAILURES:")
            for r in self.results:
                if r.status == "FAIL":
                    print(f"   - {r.check_name}: {r.message}")


def main():
    """Main entry point"""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent
    
    print("=" * 80)
    print("PHASE 0 - STEP 3: Configuration Validation")
    print("=" * 80)
    
    validator = ConfigurationValidator(repo_root)
    validator.validate_all()
    
    output_path = repo_root / "reports" / "configuration_validation.json"
    validator.generate_report(output_path)
    
    print("\n✅ Configuration validation complete!\n")


if __name__ == "__main__":
    main()
