"""
Script to remove unicode characters and AI markers from all project files.
Cleans up checkmarks, crosses, and other AI-generated status indicators.
"""

import re
from pathlib import Path

# Unicode characters and AI markers to completely remove
UNICODE_REMOVALS = [
    '', '', '', '',  # Checkmarks
    '', '', '', '',  # Crosses/rejections
    '', '', '', '',  # Warnings
    '', '', '', '',  # Progress/time
    '', '', '',      # Settings/tools
    '', '', '',       # Documents
    '', '', '',       # Highlights
    '', '', '',       # Analysis
    '', '', '',         # Special markers
]

# Text patterns to clean (AI status markers)
TEXT_PATTERNS = [
    r'\[DONE\]\s*',
    r'\[OK\]\s*',
    r'\[COMPLETED\]\s*',
    r'\[IN PROGRESS\]\s*',
    r'\[FAILED\]\s*',
    r'\[FAIL\]\s*',
    r'\[WARNING\]\s*',
]

def clean_unicode(text):
    """Remove unicode characters and AI status markers."""
    # Remove unicode symbols
    for symbol in UNICODE_REMOVALS:
        text = text.replace(symbol, '')
    
    # Remove text patterns
    for pattern in TEXT_PATTERNS:
        text = re.sub(pattern, '', text)
    
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
