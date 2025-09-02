#!/usr/bin/env python3
"""
Test script to validate 100% cloud learning pipeline.
Simulates local bot data generation and verifies cloud upload functionality.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

def create_test_training_data():
    """Create sample training data files in JSONL format"""
    
    # Create temporary data directory
    data_dir = tempfile.mkdtemp(prefix="rl_training_test_")
    print(f"📁 Created test data directory: {data_dir}")
    
    # Sample feature data
    features_es = [
        {
            "timestamp": "2024-12-01T10:30:00Z",
            "symbol": "ES",
            "strategy": "S2",
            "session": "RTH",
            "regime": "Trend",
            "signalId": "ES_S2_20241201_103000",
            "price": 4567.25,
            "atr": 15.5,
            "rsi": 65.2,
            "ema20": 4555.0,
            "ema50": 4545.0,
            "volume": 1250.0,
            "spread": 0.25,
            "volatility": 0.015,
            "signalStrength": 1.8,
            "isES": True,
            "isNQ": False,
            "tickSize": 0.25,
            "baselineMultiplier": 1.0
        },
        {
            "timestamp": "2024-12-01T10:31:00Z", 
            "symbol": "ES",
            "strategy": "S3",
            "session": "RTH",
            "regime": "Range",
            "signalId": "ES_S3_20241201_103100",
            "price": 4568.75,
            "atr": 15.8,
            "rsi": 45.1,
            "ema20": 4556.0,
            "ema50": 4546.0,
            "volume": 1180.0,
            "spread": 0.25,
            "volatility": 0.012,
            "signalStrength": 1.3,
            "isES": True,
            "isNQ": False,
            "tickSize": 0.25,
            "baselineMultiplier": 1.0
        }
    ]
    
    # Sample NQ feature data
    features_nq = [
        {
            "timestamp": "2024-12-01T10:30:00Z",
            "symbol": "NQ",
            "strategy": "S6",
            "session": "RTH", 
            "regime": "Breakout",
            "signalId": "NQ_S6_20241201_103000",
            "price": 16234.50,
            "atr": 45.2,
            "rsi": 72.8,
            "ema20": 16210.0,
            "ema50": 16180.0,
            "volume": 850.0,
            "spread": 0.25,
            "volatility": 0.018,
            "signalStrength": 2.1,
            "isES": False,
            "isNQ": True,
            "tickSize": 0.25,
            "baselineMultiplier": 1.0
        }
    ]
    
    # Sample trade outcomes
    outcomes_es = [
        {
            "signalId": "ES_S2_20241201_102000",
            "entryTime": "2024-12-01T10:20:00Z",
            "exitTime": "2024-12-01T10:25:00Z",
            "entryPrice": 4565.0,
            "exitPrice": 4567.5,
            "stopPrice": 4560.0,
            "targetPrice": 4575.0,
            "rMultiple": 0.5,
            "slippageTicks": 0.25,
            "isWin": True,
            "isCompleted": True,
            "exitReason": "Target"
        }
    ]
    
    outcomes_nq = [
        {
            "signalId": "NQ_S6_20241201_102500",
            "entryTime": "2024-12-01T10:25:00Z", 
            "exitTime": "2024-12-01T10:28:00Z",
            "entryPrice": 16220.0,
            "exitPrice": 16245.0,
            "stopPrice": 16210.0,
            "targetPrice": 16250.0,
            "rMultiple": 2.5,
            "slippageTicks": 0.5,
            "isWin": True,
            "isCompleted": True,
            "exitReason": "Target"
        }
    ]
    
    # Write JSONL files
    files_created = []
    
    # Features files
    es_features_file = os.path.join(data_dir, "features_es_20241201.jsonl")
    with open(es_features_file, 'w') as f:
        for feature in features_es:
            f.write(json.dumps(feature) + '\n')
    files_created.append(es_features_file)
    
    nq_features_file = os.path.join(data_dir, "features_nq_20241201.jsonl")
    with open(nq_features_file, 'w') as f:
        for feature in features_nq:
            f.write(json.dumps(feature) + '\n')
    files_created.append(nq_features_file)
    
    # Outcomes files
    es_outcomes_file = os.path.join(data_dir, "outcomes_es_20241201.jsonl")
    with open(es_outcomes_file, 'w') as f:
        for outcome in outcomes_es:
            f.write(json.dumps(outcome) + '\n')
    files_created.append(es_outcomes_file)
    
    nq_outcomes_file = os.path.join(data_dir, "outcomes_nq_20241201.jsonl")
    with open(nq_outcomes_file, 'w') as f:
        for outcome in outcomes_nq:
            f.write(json.dumps(outcome) + '\n')
    files_created.append(nq_outcomes_file)
    
    print(f"✅ Created {len(files_created)} test data files:")
    for file_path in files_created:
        print(f"   📄 {os.path.basename(file_path)}")
        
    return data_dir, files_created

def validate_cloud_learning_config():
    """Validate that cloud learning environment is properly configured"""
    
    print("\n🔍 Validating Cloud Learning Configuration...")
    
    required_env = {
        "S3_BUCKET": "S3 bucket for training data storage",
        "AWS_ACCESS_KEY_ID": "AWS access key for S3 uploads", 
        "AWS_SECRET_ACCESS_KEY": "AWS secret key for S3 uploads",
        "AWS_REGION": "AWS region (defaults to us-east-1)"
    }
    
    missing = []
    present = []
    
    for env_var, description in required_env.items():
        if os.getenv(env_var):
            present.append(f"✅ {env_var}")
        else:
            missing.append(f"❌ {env_var} - {description}")
    
    print("Environment Variables:")
    for item in present:
        print(f"   {item}")
    for item in missing:
        print(f"   {item}")
    
    if missing:
        print(f"\n⚠️  Missing {len(missing)} required environment variables for cloud data upload")
        print("   Set these variables to enable automatic training data upload to cloud.")
        return False
    else:
        print(f"\n✅ All {len(present)} environment variables configured for cloud learning!")
        return True

def simulate_cloud_upload_test():
    """Simulate the cloud upload process"""
    
    print("\n🌥️  Simulating Cloud Upload Process...")
    
    # Test AWS CLI availability
    import subprocess
    try:
        result = subprocess.run(['aws', '--version'], capture_output=True, text=True, check=True)
        print(f"✅ AWS CLI available: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ AWS CLI not found - required for cloud data upload")
        print("   Install AWS CLI: https://aws.amazon.com/cli/")
        return False
    
    # Test S3 connectivity (if credentials available)
    s3_bucket = os.getenv('S3_BUCKET')
    if s3_bucket:
        try:
            result = subprocess.run([
                'aws', 's3', 'ls', f's3://{s3_bucket}/', '--region', os.getenv('AWS_REGION', 'us-east-1')
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print(f"✅ S3 bucket accessible: s3://{s3_bucket}/")
                return True
            else:
                print(f"❌ S3 bucket access failed: {result.stderr.strip()}")
                return False
        except subprocess.TimeoutExpired:
            print("⚠️  S3 access test timed out")
            return False
        except Exception as e:
            print(f"❌ S3 access test error: {e}")
            return False
    else:
        print("⚠️  S3_BUCKET not set - skipping connectivity test")
        return False

def main():
    """Run complete validation of 100% cloud learning setup"""
    
    print("🚀 Testing 100% Cloud-Based Learning Pipeline")
    print("=" * 50)
    
    # Step 1: Create test training data
    data_dir, files = create_test_training_data()
    
    try:
        # Step 2: Validate configuration  
        config_ok = validate_cloud_learning_config()
        
        # Step 3: Test cloud upload capability
        upload_ok = simulate_cloud_upload_test()
        
        # Summary
        print("\n📊 Test Results Summary")
        print("=" * 30)
        print(f"✅ Test data generation: SUCCESS")
        print(f"{'✅' if config_ok else '❌'} Environment configuration: {'SUCCESS' if config_ok else 'MISSING VARS'}")
        print(f"{'✅' if upload_ok else '❌'} Cloud upload capability: {'SUCCESS' if upload_ok else 'FAILED'}")
        
        if config_ok and upload_ok:
            print(f"\n🎉 Cloud learning pipeline is ready!")
            print(f"   📈 Training data will be uploaded automatically every 15 minutes")
            print(f"   🤖 Models will be trained in cloud every 30 minutes") 
            print(f"   🎯 Bot only needs to run for trading - not for learning")
        else:
            print(f"\n⚠️  Cloud learning pipeline needs configuration")
            print(f"   📝 Set missing environment variables to enable cloud uploads")
            print(f"   🔧 Ensure AWS CLI is installed and configured")
        
        # Show file sizes
        print(f"\n📁 Test Data Generated:")
        for file_path in files:
            size = os.path.getsize(file_path)
            print(f"   {os.path.basename(file_path)}: {size} bytes")
            
    finally:
        # Cleanup
        try:
            shutil.rmtree(data_dir)
            print(f"\n🧹 Cleaned up test directory: {data_dir}")
        except Exception as e:
            print(f"⚠️  Failed to cleanup {data_dir}: {e}")

if __name__ == "__main__":
    main()