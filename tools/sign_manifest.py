#!/usr/bin/env python3
"""
Manifest HMAC signature generator for secure model updates.

Generates HMAC-SHA256 signatures for model manifest JSON files to ensure
integrity and authenticity during 24/7 cloud learning updates.

Usage:
    python tools/sign_manifest.py --manifest models/current.json --key $MANIFEST_HMAC_KEY
"""

import argparse
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path


def generate_manifest_signature(manifest_path: str, hmac_key: str) -> str:
    """
    Generate HMAC-SHA256 signature for manifest file.
    
    Args:
        manifest_path: Path to manifest JSON file
        hmac_key: Secret key for HMAC generation
        
    Returns:
        Hex-encoded HMAC signature
    """
    try:
        # Read and normalize manifest content
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)
        
        # Create canonical JSON representation (sorted keys, no whitespace)
        canonical_json = json.dumps(manifest_data, sort_keys=True, separators=(',', ':'))
        canonical_bytes = canonical_json.encode('utf-8')
        
        # Generate HMAC-SHA256 signature
        hmac_obj = hmac.new(
            hmac_key.encode('utf-8'),
            canonical_bytes,
            hashlib.sha256
        )
        
        return hmac_obj.hexdigest()
        
    except FileNotFoundError:
        raise ValueError(f"Manifest file not found: {manifest_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in manifest: {e}")
    except Exception as e:
        raise ValueError(f"Failed to generate signature: {e}")


def verify_manifest_signature(manifest_path: str, hmac_key: str, expected_sig: str) -> bool:
    """
    Verify HMAC signature for manifest file.
    
    Args:
        manifest_path: Path to manifest JSON file
        hmac_key: Secret key for HMAC verification
        expected_sig: Expected signature to verify against
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        actual_sig = generate_manifest_signature(manifest_path, hmac_key)
        return hmac.compare_digest(actual_sig, expected_sig)
    except Exception:
        return False


def add_signature_to_manifest(manifest_path: str, signature: str) -> None:
    """
    Add HMAC signature to manifest file.
    
    Args:
        manifest_path: Path to manifest JSON file
        signature: HMAC signature to add
    """
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)
        
        # Add signature and metadata
        manifest_data['signature'] = {
            'algorithm': 'HMAC-SHA256',
            'value': signature,
            'signed_at': json.dumps({
                'timestamp': 'GENERATED_BY_GITHUB_ACTIONS',
                'version': manifest_data.get('version', 'unknown')
            })
        }
        
        # Write back with proper formatting
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2, sort_keys=True)
            
    except Exception as e:
        raise ValueError(f"Failed to add signature to manifest: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate and verify HMAC signatures for model manifests'
    )
    parser.add_argument(
        '--manifest', 
        required=True,
        help='Path to manifest JSON file'
    )
    parser.add_argument(
        '--key',
        help='HMAC key (or set MANIFEST_HMAC_KEY env var)'
    )
    parser.add_argument(
        '--verify',
        help='Verify signature instead of generating'
    )
    parser.add_argument(
        '--add-to-manifest',
        action='store_true',
        help='Add generated signature to manifest file'
    )
    parser.add_argument(
        '--output-only',
        action='store_true',
        help='Only output signature, do not modify files'
    )
    
    args = parser.parse_args()
    
    # Get HMAC key from args or environment
    hmac_key = args.key or os.environ.get('MANIFEST_HMAC_KEY')
    if not hmac_key:
        print("ERROR: HMAC key required via --key or MANIFEST_HMAC_KEY env var", file=sys.stderr)
        sys.exit(1)
    
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if args.verify:
            # Verify existing signature
            is_valid = verify_manifest_signature(str(manifest_path), hmac_key, args.verify)
            print(f"Signature verification: {'VALID' if is_valid else 'INVALID'}")
            sys.exit(0 if is_valid else 1)
        else:
            # Generate new signature
            signature = generate_manifest_signature(str(manifest_path), hmac_key)
            
            if args.output_only:
                print(signature)
            else:
                print(f"Generated signature: {signature}")
                
                if args.add_to_manifest:
                    add_signature_to_manifest(str(manifest_path), signature)
                    print(f"Added signature to manifest: {manifest_path}")
    
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
