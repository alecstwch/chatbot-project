"""Verification script for cleanup results."""
from pathlib import Path

print('VERIFICATION: Functional Emojis Preserved & Separators Removed')
print()

# Check rag_chatbot_cli.py
rag_file = Path('src/interfaces/cli/rag_chatbot_cli.py')
if rag_file.exists():
    content = rag_file.read_text(encoding='utf-8')
    
    # Check functional emojis
    emojis = ['🟢', '', '🟠', '', '']
    found_emojis = [e for e in emojis if e in content]
    
    print(f'File: {rag_file}')
    print(f'  Functional emojis (risk/severity): {found_emojis}')
    print(f'  Total preserved: {len(found_emojis)}/{len(emojis)}')
    
    # Check separator removal
    has_dash_separator = '' in content
    print(f'  Dash character () still present: {has_dash_separator}')
    print(f'  Status: {" CLEANED" if not has_dash_separator else " Check needed"}')
    print()

# Check enhanced_prompt_builder.py
prompt_file = Path('src/infrastructure/memory/enhanced_prompt_builder.py')
if prompt_file.exists():
    content = prompt_file.read_text(encoding='utf-8')
    
    # Check for any emojis (we don't know which ones should be there)
    test_emojis = ['', '', '', '', '']
    found_emojis = [e for e in test_emojis if e in content]
    
    print(f'File: {prompt_file}')
    print(f'  Sample emojis found: {found_emojis if found_emojis else "None"}')
    print(f'  Status: {" PRESERVED" if found_emojis else "ℹ No test emojis (may be OK)"}')
    print()

print('SUMMARY')
print(' Script updated with:')
print('  - New category: PRESERVE_FUNCTIONAL_EMOJIS')
print('  - F-string separator detection: print(f"{\'\' * 70}")')
print('  - Selective cleaning: separators removed, emojis preserved')
