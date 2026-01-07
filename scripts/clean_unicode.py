"""
Script to remove unicode characters and AI markers from all project files.
"""

import re
from pathlib import Path

# Unicode characters to remove (map to plain text)
UNICODE_REPLACEMENTS = {
    '[DONE]': '[DONE]',
    '[OK]': '[OK]',
    '': '',
    '': '',
    '': '',
    '': '',
    '[IN PROGRESS]': '[IN PROGRESS]',
    '': '',
    '[FAILED]': '[FAILED]',
    '[FAIL]': '[FAIL]',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '[WARNING]': '[WARNING]',
}

# Patterns that might indicate AI generation (be careful not to break code)
AI_MARKERS_TO_REMOVE = [
    # These are safe to remove from comments/docs
]

def clean_unicode(text):
    """Replace unicode characters with plain text equivalents."""
    for unicode_char, replacement in UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, replacement)
    return text

def clean_file(file_path):
    """Clean a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original = content
        content = clean_unicode(content)
        
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Clean all markdown and Python files."""
    project_root = Path(__file__).parent.parent
    
    # Patterns to include
    patterns = ['**/*.md', '**/*.py', '**/*.ps1', '**/*.txt', '**/*.yml', '**/*.yaml']
    
    # Directories to exclude
    exclude_dirs = {
        'chatbot-env', 'marker-env', '.pytest_cache', 
        'htmlcov', '__pycache__', '.egg-info', 'build', 'dist'
    }
    
    files_changed = 0
    files_processed = 0
    
    for pattern in patterns:
        for file_path in project_root.glob(pattern):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue
            
            files_processed += 1
            if clean_file(file_path):
                files_changed += 1
                print(f"Cleaned: {file_path.relative_to(project_root)}")
    
    print(f"\nProcessed {files_processed} files, changed {files_changed} files")

if __name__ == "__main__":
    main()
