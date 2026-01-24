"""
Script to remove unicode characters and AI markers from all project files.
Cleans up checkmarks, crosses, decorative separators, emojis, and other AI-generated patterns.
Extended to detect print statements with repeated characters, hard-coded separators,
and various AI code generation patterns.
"""

import re
from pathlib import Path
import mimetypes
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    print("Warning: python-magic not installed. Using fallback binary detection.")
    print("Install with: pip install python-magic python-magic-bin")

# ============================================================================
# CONFIGURATION: Files and directories to skip from cleaning
# ============================================================================

# Files with functional emojis (sentiment analysis, UI elements, etc.)
# These files use emojis for legitimate functional purposes, not decoration.
# They will only be cleaned for separator patterns, but emojis will be preserved.
PRESERVE_FUNCTIONAL_EMOJIS = [
]

# Files to completely skip from any cleaning
SKIP_FILES = [
    'clean_unicode.py',  # Always skip this script itself
    'ENHANCED_RAG_CHATBOT.md',
    'RAG_SETUP.md',
    'test_text_preprocessor.py',
]

SKIP_DIRECTORIES = [
    'research_papers',  # Skip all research papers documentation
]

# ============================================================================
# UNICODE AND EMOJI PATTERNS
# ============================================================================

# Comprehensive emoji and unicode symbol removal using regex patterns
# This covers all emoji ranges in Unicode
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
    "\U00002600-\U000026FF"  # miscellaneous symbols
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F018-\U0001F270"  # various symbols
    "\u200d"                 # zero width joiner
    "\ufe0f"                 # variation selector
    "]+", 
    flags=re.UNICODE
)

# Common unicode symbols used by AI (checkmarks, arrows, bullets, etc.)
COMMON_SYMBOLS = [
    '✓', '✔', '✅', '✗', '✘', '❌',  # Checkmarks and crosses
    '⚠', '⚡', '❗', '❓', '⭐', '🔥',  # Warnings and highlights
    '📝', '📋', '📚', '📖', '📄', '📃',  # Documents
    '🔧', '⚙', '🛠', '⚡',  # Tools/settings
    '👤', '👥', '🤖', '💬', '💭',  # People and chat
    '➡', '⬅', '⬆', '⬇', '↔', '↕', '→',  # Arrows (added →)
    '•', '▪', '▫', '●', '○',  # Bullets (removed ◦ - will be replaced with *)
    '⏰', '⏱', '⌚', '🕐', '⏭',  # Time (added ⏭)
    '✨', '🌟', '💡', '🎯', '🟡',  # Highlights (added 🟡)
]

# Character replacements (before removal)
CHARACTER_REPLACEMENTS = {
    '◦': '*',  # Replace hollow bullet with asterisk
}


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

# AI-generated separator patterns (decorative lines)
# These patterns detect hard-coded separator lines with repeated characters
SEPARATOR_PATTERNS = [
    # Python/Shell/Config comment separators with 10+ repeated characters
    r'^\s*#\s*[=]{10,}\s*$',        # 
    r'^\s*#\s*[-]{10,}\s*$',        # 
    r'^\s*#\s*[*]{10,}\s*$',        # # ********************
    r'^\s*#\s*[~]{10,}\s*$',        # # ~~~~~~~~~~~~~~~~~~~~
    r'^\s*#\s*[_]{10,}\s*$',        # # ____________________
    
    # Multi-line comment separators (/* ... */ style)
    r'^\s*/\*\s*[=*-]{10,}\s*\*/',  # /* ================== */
    
    # HTML/Markdown comment separators
    r'^\s*<!--\s*[=\-*]{10,}\s*-->',  # <!-- ============== -->
    
    # Plain separators (in text/config files)
    r'^\s*[=]{20,}\s*$',            
    r'^\s*[-]{20,}\s*$',            
    r'^\s*[*]{20,}\s*$',            # ********************
]

# Code patterns that print/generate separator lines - these match COMPLETE LINES only
# These detect lines that ONLY print separators, not separator patterns within larger expressions
CODE_SEPARATOR_PATTERNS = [
    # Python print statements - must be complete line with only the separator print
    r'^\s*print\s*\(\s*["\'][=\-*~_#]+["\']\s*\*\s*\d+\s*\)\s*$',  # print("="*50)
    r'^\s*print\s*\(\s*\d+\s*\*\s*["\'][=\-*~_#]+["\']\s*\)\s*$',  # print(50*"=")
    r'^\s*print\s*\(\s*["\'][=\-*~_#]{20,}["\']\s*\)\s*$',  # print("====================")
    
    # Python f-string separators with character multiplication - complete lines only
    # Handles both plain f-strings and those with \n prefix
    r'^\s*print\s*\(\s*f["\']\\n\{["\'][=\-*~_#─┌└│┤├┬┴]+["\']\s*\*\s*\d+\}["\']\s*\)\s*$',  # print(f"\n{'─' * 70}")
    r'^\s*print\s*\(\s*f["\']\{["\'][=\-*~_#─┌└│┤├┬┴]+["\']\s*\*\s*\d+\}["\']\s*\)\s*$',  # print(f"{'─' * 70}")
    r'^\s*print\s*\(\s*f["\']\{\d+\s*\*\s*["\'][=\-*~_#─┌└│┤├┬┴]+["\']\}["\']\s*\)\s*$',  # print(f"{70 * '─'}")
    
    # Python logger/console with separators - complete lines only
    r'^\s*(?:logger|logging)\.\w+\s*\(\s*["\'][=\-*~_#]+["\']\s*\*\s*\d+\s*\)\s*$',
    
    # JavaScript/TypeScript console statements - complete lines only
    r'^\s*console\.\w+\s*\(\s*["\'][=\-*~_#]+["\']\s*\.repeat\s*\(\s*\d+\s*\)\s*\)\s*;?\s*$',
    
    # Echo commands in shell scripts - complete lines only
    r'^\s*echo\s+["\'][=\-*~_#]{20,}["\']\s*$',
]

# AI-generated comment patterns
AI_COMMENT_PATTERNS = [
    # Generic AI placeholder comments
    r'#\s*TODO:\s+Implement\s+this',
    r'#\s*TODO:\s+Add\s+',
    r'#\s*FIXME:\s+',
    r'#\s*XXX:\s+',
    r'#\s*HACK:\s+',
    
    # Overly verbose AI documentation patterns
    r'#\s*={3,}\s*$',  # === (shorter separator)
    r'#\s*-{3,}\s*$',  # --- (shorter separator)
]

# Binary file extensions to skip
BINARY_EXTENSIONS = {
    '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib', '.exe',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
    '.pdf', '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',
    '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac',
    '.ttf', '.otf', '.woff', '.woff2', '.eot',
    '.db', '.sqlite', '.sqlite3',
    '.pkl', '.pickle', '.npy', '.npz',
    '.pt', '.pth', '.h5', '.hdf5',
    '.docx', '.xlsx', '.pptx',
}

# File extensions to process
PROCESSABLE_EXTENSIONS = {
    # Code files
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
    
    # Config files
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.xml', '.properties', '.env',
    
    # Shell scripts
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
    
    # Documentation
    '.md', '.rst', '.txt', '.tex',
    
    # Web files
    '.html', '.css', '.scss', '.sass', '.less',
    
    # Data files
    '.csv', '.tsv', '.sql',
}


def is_binary_file(file_path):
    """Check if a file is binary using python-magic or fallback methods."""
    # Check by extension first (quick check)
    if file_path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    
    # Use python-magic if available (most reliable)
    if HAS_MAGIC:
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(str(file_path))
            
            # Text files or specific known text types
            if mime_type.startswith('text/'):
                return False
            if mime_type in ('application/json', 'application/xml', 'application/x-yaml',
                           'application/javascript', 'application/x-sh', 'application/x-python'):
                return False
            
            # Everything else is likely binary
            return True
        except Exception as e:
            # Fall through to fallback method
            pass
    
    # Fallback: Check by MIME type (less reliable)
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type and not mime_type.startswith('text/'):
        return True
    
    # Fallback: Try to read first 8KB to detect binary content
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(8192)
            # If null bytes present, likely binary
            if b'\x00' in chunk:
                return True
    except Exception:
        return True
    
    return False


def is_processable_file(file_path):
    """Check if a file should be processed."""
    # Skip binary files
    if is_binary_file(file_path):
        return False
    
    # Process files with known text extensions
    ext = file_path.suffix.lower()
    if ext in PROCESSABLE_EXTENSIONS:
        return True
    
    # Also process files with no extension if they're text
    if not ext:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Try to read some content
            return True
        except (UnicodeDecodeError, Exception):
            return False
    
    return False


def clean_unicode(text):
    """Remove emoji, unicode symbols, and AI status markers."""
    # Apply character replacements first (before removal)
    for old_char, new_char in CHARACTER_REPLACEMENTS.items():
        text = text.replace(old_char, new_char)
    
    # Remove all emojis using the comprehensive emoji pattern
    text = EMOJI_PATTERN.sub('', text)
    
    # Remove common unicode symbols
    for symbol in COMMON_SYMBOLS:
        text = text.replace(symbol, '')
    
    # Remove text patterns (AI status markers)
    for pattern in TEXT_PATTERNS:
        text = re.sub(pattern, '', text)
    
    return text


def clean_separators(text):
    """Remove AI-generated separator lines and code patterns."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        should_skip = False
        
        # Check if line matches hard-coded separator patterns
        for pattern in SEPARATOR_PATTERNS:
            if re.match(pattern, line):
                should_skip = True
                break
        
        # Check if line matches code separator patterns (complete lines only)
        if not should_skip:
            for pattern in CODE_SEPARATOR_PATTERNS:
                if re.match(pattern, line):
                    should_skip = True
                    break
        
        # Check if line matches AI comment patterns
        if not should_skip:
            for pattern in AI_COMMENT_PATTERNS:
                if re.match(pattern, line):
                    should_skip = True
                    break
        
        # Keep line if it doesn't match any pattern
        if not should_skip:
            cleaned_lines.append(line)
    
    result = '\n'.join(cleaned_lines)
    return result


def clean_file(file_path, verbose=False):
    """Clean a single file."""
    try:
        # Check if file should be skipped (by filename)
        if file_path.name in SKIP_FILES:
            if verbose:
                print(f"Skipping configured file: {file_path.name}")
            return False
        
        # Check if file is in a skipped directory
        for skip_dir in SKIP_DIRECTORIES:
            if skip_dir in file_path.parts:
                if verbose:
                    print(f"Skipping file in {skip_dir}: {file_path.name}")
                return False
        
        # Check if file has functional emojis (special handling)
        preserve_emojis = file_path.name in PRESERVE_FUNCTIONAL_EMOJIS
        if preserve_emojis and verbose:
            print(f"Preserving functional emojis in: {file_path.name}")
        
        # Check if file should be processed
        if not is_processable_file(file_path):
            if verbose:
                print(f"Skipping binary/unprocessable: {file_path.name}")
            return False
        
        # Read file content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        original = content
        
        # Apply cleaning operations based on file type
        if preserve_emojis:
            # Only clean separators, preserve all unicode/emojis
            content = clean_separators(content)
        else:
            # Apply all cleaning operations
            content = clean_unicode(content)
            content = clean_separators(content)
        
        # Remove multiple consecutive blank lines (AI often adds these)
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        
        # Write back if changed
        if content != original:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Clean all text files in the entire project."""
    import sys
    
    project_root = Path(__file__).parent.parent
    
    # Directories to exclude
    exclude_dirs = {
        'chatbot-env', 'marker-env', '.pytest_cache', 
        'htmlcov', '__pycache__', '.egg-info', 'build', 'dist',
        'node_modules', '.git', '.venv', 'venv', '.tox',
        'models', 'data',  # Exclude model and data directories
        'NLP_Paper_Template',  # Exclude LaTeX template
    }
    
    files_changed = 0
    files_processed = 0
    files_skipped = 0
    
    print("Starting AI pattern cleanup...")
    print(f"Project root: {project_root}")
    print("Processing: ENTIRE PROJECT")
    print(f"Excluded directories: {', '.join(sorted(exclude_dirs))}\n")
    
    sep = "="*60
    print(f"{sep}")
    print("Scanning and cleaning files...")
    print(sep)
    
    # Walk through all files in the project
    for file_path in project_root.rglob('*'):
        # Skip directories
        if not file_path.is_file():
            continue
        
        # Skip excluded directories
        if any(excluded in file_path.parts for excluded in exclude_dirs):
            continue
        
        files_processed += 1
        
        # Clean the file
        try:
            changed = clean_file(file_path, verbose=False)
            if changed:
                files_changed += 1
                rel_path = file_path.relative_to(project_root)
                print(f"   Cleaned: {rel_path}")
        except Exception as e:
            files_skipped += 1
            print(f"   Error processing {file_path.name}: {e}")
    
    # Summary
    sep = "="*60
    print(f"\n{sep}")
    print("CLEANUP SUMMARY")
    print(sep)
    print(f"Files processed: {files_processed}")
    print(f"Files cleaned:   {files_changed}")
    print(f"Files skipped:   {files_skipped}")
    print(f"Files unchanged: {files_processed - files_changed - files_skipped}")
    print(f"\nAI pattern cleanup complete!")


if __name__ == "__main__":
    main()
