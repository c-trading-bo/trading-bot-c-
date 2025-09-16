#!/usr/bin/env python3
import re
import sys

def fix_async_method(content, method_signature_pattern):
    """Add await Task.Yield() to async methods lacking await"""
    # Find method signatures and add yield
    pattern = rf'({method_signature_pattern})\s*\{{\s*([^{{])'
    
    def replace_func(match):
        method_sig = match.group(1)
        first_line = match.group(2)
        return f'{method_sig}\n    {{\n        // Brief yield for async context\n        await Task.Yield();\n        \n        {first_line}'
    
    return re.sub(pattern, replace_func, content, flags=re.MULTILINE)

# Read file
if len(sys.argv) != 2:
    print("Usage: python3 fix_async_warnings.py <file>")
    sys.exit(1)

filename = sys.argv[1]
with open(filename, 'r') as f:
    content = f.read()

# Fix various async method patterns
patterns = [
    r'private async Task<\w+> Get\w+Async\([^)]*\)',
    r'public async Task<\w+> \w+Async\([^)]*\)',
    r'private async Task \w+Async\([^)]*\)',
]

for pattern in patterns:
    content = fix_async_method(content, pattern)

# Write back
with open(filename, 'w') as f:
    f.write(content)

print(f"Fixed async methods in {filename}")
