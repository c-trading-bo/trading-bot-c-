#!/usr/bin/env python3
"""
Script to batch fix CA1031 violations by replacing generic Exception catches
with more specific exception types for common patterns.
"""

import re
import os
import sys

def fix_file_operation_exceptions(content):
    """Replace generic Exception catches with specific file operation exceptions."""
    
    # Pattern for file operation exception handling
    pattern = r'(\s*)catch \(Exception (.*?)\)\s*\n(\s*)\{\s*\n(.*?)\s*\n(\s*)\}'
    
    def replace_exception(match):
        indent = match.group(1)
        ex_var = match.group(2)
        brace_indent = match.group(3)
        content_line = match.group(4)
        closing_indent = match.group(5)
        
        # Determine appropriate exception types based on content
        if any(word in content_line.lower() for word in ['file', 'directory', 'path', 'delete', 'write', 'read']):
            return f"""{indent}catch (UnauthorizedAccessException {ex_var})
{brace_indent}{{
{content_line}
{closing_indent}}}
{indent}catch (System.IO.IOException {ex_var})
{brace_indent}{{
{content_line}
{closing_indent}}}
{indent}catch (DirectoryNotFoundException {ex_var})
{brace_indent}{{
{content_line}
{closing_indent}}}"""
        elif 'json' in content_line.lower() or 'serialize' in content_line.lower():
            return f"""{indent}catch (JsonException {ex_var})
{brace_indent}{{
{content_line}
{closing_indent}}}
{indent}catch (ArgumentException {ex_var})
{brace_indent}{{
{content_line}
{closing_indent}}}
{indent}catch (InvalidOperationException {ex_var})
{brace_indent}{{
{content_line}
{closing_indent}}}"""
        else:
            # Generic fallback
            return f"""{indent}catch (ArgumentException {ex_var})
{brace_indent}{{
{content_line}
{closing_indent}}}
{indent}catch (InvalidOperationException {ex_var})
{brace_indent}{{
{content_line}
{closing_indent}}}
{indent}catch (System.IO.IOException {ex_var})
{brace_indent}{{
{content_line}
{closing_indent}}}"""
    
    return re.sub(pattern, replace_exception, content, flags=re.MULTILINE | re.DOTALL)

def process_file(filepath):
    """Process a single file to fix CA1031 violations."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        content = fix_file_operation_exceptions(content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        else:
            print(f"No changes: {filepath}")
            return False
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_ca1031_batch.py <file1> [file2] ...")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        if os.path.exists(filepath):
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()