#!/usr/bin/env python3
import re

def find_true_key_source():
    with open('.github/workflows/cloud_bot_mechanic.yml', 'r', encoding='utf-8') as f:
        content = f.read()

    print("🔍 SEARCHING FOR TRUE KEY SOURCE...")
    
    # Look for unquoted boolean values that could become True keys
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Check for various patterns that could create a True key
        patterns = [
            r':\s*true\s*$',     # ends with unquoted true
            r':\s*True\s*$',     # ends with unquoted True
            r'runs-on.*ubuntu',  # runs-on statements
            r'required:\s*false\s*$',  # required: false
            r'required:\s*true\s*$',   # required: true
        ]
        
        for pattern in patterns:
            if re.search(pattern, line):
                print(f"Line {i}: {repr(line)} - Pattern: {pattern}")

if __name__ == "__main__":
    find_true_key_source()
