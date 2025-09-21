#!/usr/bin/env python3
"""
Comprehensive verification of configuration integration for the newly added:
1. Neutral confidence parameters
2. Autonomous resource management environment variables
"""

import json
import os
import sys

def verify_json_configuration():
    """Verify appsettings.json has correct neutral confidence parameters"""
    print("🔍 Verifying appsettings.json configuration...")
    
    config_path = "src/UnifiedOrchestrator/appsettings.json"
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Navigate to confidence config
        confidence_config = config.get("IntelligenceStack", {}).get("ml", {}).get("confidence", {})
        
        # Check required parameters
        checks = [
            ("enabled", True),
            ("minConfidence", 0.52),
            ("neutralConfidenceMin", 0.45),
            ("neutralConfidenceMax", 0.50),
            ("kellyClip", 0.35),
            ("edgeConversionOffset", 0.5),
            ("edgeConversionMultiplier", 2.0),
            ("confidenceMultiplierOffset", 0.5),
            ("confidenceMultiplierScale", 4.0)
        ]
        
        all_passed = True
        for key, expected_value in checks:
            actual_value = confidence_config.get(key)
            if actual_value != expected_value:
                print(f"❌ {key}: expected {expected_value}, got {actual_value}")
                all_passed = False
            else:
                print(f"✅ {key}: {actual_value}")
        
        # Validate neutral confidence range
        neutral_min = confidence_config.get("neutralConfidenceMin", 0)
        neutral_max = confidence_config.get("neutralConfidenceMax", 0)
        min_conf = confidence_config.get("minConfidence", 0)
        
        if neutral_min >= neutral_max:
            print(f"❌ Invalid neutral range: neutralConfidenceMin ({neutral_min}) >= neutralConfidenceMax ({neutral_max})")
            all_passed = False
        elif neutral_max >= min_conf:
            print(f"❌ Invalid range: neutralConfidenceMax ({neutral_max}) >= minConfidence ({min_conf})")
            all_passed = False
        else:
            print(f"✅ Valid neutral confidence range: {neutral_min} - {neutral_max} (below minConfidence {min_conf})")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading {config_path}: {e}")
        return False

def verify_environment_configuration():
    """Verify .env file has correct autonomous resource management variables"""
    print("\n🔍 Verifying .env configuration...")
    
    env_path = ".env"
    
    try:
        with open(env_path, 'r') as f:
            env_content = f.read()
        
        # Check required environment variables
        required_vars = [
            "LEARNING_RESOURCE_LIMIT_ENABLED=true",
            "LEARNING_CPU_LIMIT_PERCENT=50",
            "LEARNING_MEMORY_LIMIT_MB=2048",
            "RESOURCE_MONITORING_ENABLED=true",
            "DEVICE_TRUST_ENABLED=true",
            "TRUSTED_DEVICE_ID=kevin-workstation-primary",
            "AUTONOMOUS_DEVICE_LOCK=true"
        ]
        
        all_passed = True
        for var in required_vars:
            if var in env_content:
                print(f"✅ {var}")
            else:
                print(f"❌ Missing: {var}")
                all_passed = False
        
        # Check sections are present
        resource_section = "# AUTONOMOUS RESOURCE MANAGEMENT"
        device_section = "# DEVICE TRUST & AUTONOMOUS FEATURES"
        
        if resource_section in env_content:
            print(f"✅ {resource_section} section found")
        else:
            print(f"❌ Missing section: {resource_section}")
            all_passed = False
            
        if device_section in env_content:
            print(f"✅ {device_section} section found")
        else:
            print(f"❌ Missing section: {device_section}")
            all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading {env_path}: {e}")
        return False

def verify_csharp_class_integration():
    """Verify ConfidenceConfig class has the new properties"""
    print("\n🔍 Verifying C# class integration...")
    
    config_class_path = "src/Abstractions/IntelligenceStackConfig.cs"
    
    try:
        with open(config_class_path, 'r') as f:
            content = f.read()
        
        # Check that ConfidenceConfig class has the new properties
        required_properties = [
            "public double NeutralConfidenceMin { get; set; } = 0.45;",
            "public double NeutralConfidenceMax { get; set; } = 0.50;"
        ]
        
        all_passed = True
        for prop in required_properties:
            if prop in content:
                print(f"✅ Found property: {prop}")
            else:
                print(f"❌ Missing property: {prop}")
                all_passed = False
        
        # Check class structure
        if "public class ConfidenceConfig" in content:
            print("✅ ConfidenceConfig class found")
        else:
            print("❌ ConfidenceConfig class not found")
            all_passed = False
            
        return all_passed
        
    except Exception as e:
        print(f"❌ Error reading {config_class_path}: {e}")
        return False

def verify_service_registration_integration():
    """Verify ConfidenceConfig is properly registered in DI"""
    print("\n🔍 Verifying service registration integration...")
    
    service_ext_path = "src/IntelligenceStack/IntelligenceStackServiceExtensions.cs"
    
    try:
        with open(service_ext_path, 'r') as f:
            content = f.read()
        
        # Check that ConfidenceConfig is registered
        confidence_registration = "services.AddSingleton<ConfidenceConfig>(provider =>"
        
        if confidence_registration in content:
            print(f"✅ ConfidenceConfig service registration found")
            
            # Check specific registration line
            ml_confidence_line = "provider.GetRequiredService<IntelligenceStackConfig>().ML.Confidence"
            if ml_confidence_line in content:
                print(f"✅ Correct ConfidenceConfig binding found")
                return True
            else:
                print(f"❌ Incorrect ConfidenceConfig binding")
                return False
        else:
            print(f"❌ ConfidenceConfig service registration not found")
            return False
            
    except Exception as e:
        print(f"❌ Error reading {service_ext_path}: {e}")
        return False

def main():
    """Run comprehensive verification"""
    print("🚀 Running comprehensive configuration verification...")
    print("=" * 70)
    
    json_ok = verify_json_configuration()
    env_ok = verify_environment_configuration()
    class_ok = verify_csharp_class_integration()
    service_ok = verify_service_registration_integration()
    
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY:")
    print(f"  📋 JSON Configuration: {'✅ PASS' if json_ok else '❌ FAIL'}")
    print(f"  🌍 Environment Configuration: {'✅ PASS' if env_ok else '❌ FAIL'}")
    print(f"  💻 C# Class Integration: {'✅ PASS' if class_ok else '❌ FAIL'}")
    print(f"  🔧 Service Registration: {'✅ PASS' if service_ok else '❌ FAIL'}")
    
    all_ok = json_ok and env_ok and class_ok and service_ok
    
    if all_ok:
        print("\n🎉 ALL VERIFICATIONS PASSED - Configuration is fully integrated and production-ready!")
        print("✅ Neutral confidence parameters properly configured")
        print("✅ Autonomous resource management properly configured")
        print("✅ C# classes properly updated")
        print("✅ Dependency injection properly wired")
        return 0
    else:
        print("\n❌ VERIFICATION FAILED - Some components are not properly configured")
        return 1

if __name__ == "__main__":
    sys.exit(main())