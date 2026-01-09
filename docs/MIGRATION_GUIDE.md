# Migration Guide - Old to New Structure

## Quick Reference: Where Did Everything Go?

### Old Location → New Location

```
OLD: src/data/preprocessor.py
NEW: src/domain/services/text_preprocessor.py (code)
     tests/unit/domain/test_text_preprocessor.py (tests)
```

## Import Changes

### Before
```python
# Old import (won't work anymore)
from src.data.preprocessor import TextPreprocessor
```

### After
```python
# New import
from src.domain.services.text_preprocessor import TextPreprocessingService
```

## Class Name Changes

### Before
```python
preprocessor = TextPreprocessor()
```

### After
```python
# Note: Class renamed for clarity
preprocessor = TextPreprocessingService()
```

## Usage Examples

### Basic Preprocessing

**Before:**
```python
from src.data.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
result = preprocessor.preprocess("Hello World!")
```

**After:**
```python
from src.domain.services.text_preprocessor import TextPreprocessingService

preprocessor = TextPreprocessingService()
result = preprocessor.preprocess("Hello World!")
```

### DataFrame Processing

**Before:**
```python
from src.data.preprocessor import preprocess_dataframe

df_processed = preprocess_dataframe(df, 'text_column')
```

**After:**
```python
# DataFrame helper moved to infrastructure layer
# For now, use service directly:
from src.domain.services.text_preprocessor import TextPreprocessingService

preprocessor = TextPreprocessingService()
df['preprocessed'] = preprocessor.batch_preprocess(df['text_column'].tolist())
```

## Configuration

### Before
```python
# Settings hardcoded in code
preprocessor = TextPreprocessor(language='english')
```

### After
```python
# Settings from environment
from config import settings

preprocessor = TextPreprocessingService(
    language=settings.preprocess_language
)

# Or use helper:
config = settings.get_preprocessing_config()
result = preprocessor.preprocess(text, **config)
```

## Testing

### Before
```python
# Tests mixed in preprocessor.py at the bottom
if __name__ == "__main__":
    # test code here
```

### After
```python
# Separate test file: tests/unit/domain/test_text_preprocessor.py
pytest tests/unit/domain/test_text_preprocessor.py
```

## Method Name Changes

Most methods unchanged, but note:

| Old Method | New Method |
|------------|------------|
| `lowercase()` | `to_lowercase()` |
| `stem_tokens()` | `apply_stemming()` |
| `lemmatize_tokens()` | `apply_lemmatization()` |

## What's New?

1. **Type-safe configuration** (config/settings.py)
2. **Environment variables** (.env)
3. **Comprehensive tests** (27 unit tests)
4. **Better documentation** (docstrings)
5. **Clean architecture** (DDD layers)

## Breaking Changes

**Class renamed**: `TextPreprocessor` → `TextPreprocessingService`  
**Import path changed**: `src.data.preprocessor` → `src.domain.services.text_preprocessor`  
**DataFrame helper**: Now manual (will be in infrastructure layer later)

## Migration Checklist

- [ ] Update all imports
- [ ] Rename class instances
- [ ] Use new configuration system
- [ ] Move tests to tests/ directory
- [ ] Copy .env.example to .env
- [ ] Run pytest to verify

## Need Help?

See:
- [README_NEW_STRUCTURE.md](README_NEW_STRUCTURE.md) - Complete guide
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Architecture details
- [RESTRUCTURING_COMPLETE.md](RESTRUCTURING_COMPLETE.md) - Summary
