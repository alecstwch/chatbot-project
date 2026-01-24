# Chatbot Project - DDD + 12-Factor Architecture

A production-ready chatbot implementation following Domain-Driven Design and 12-Factor App principles.

## Architecture

This project follows **Domain-Driven Design (DDD)** with clean separation of concerns and **12-Factor App** principles for cloud-ready deployment.

### Layer Structure

```
src/
 domain/           # Core business logic (pure, no dependencies)
 application/      # Use cases and workflows
 infrastructure/   # External concerns (ML, DB, APIs)
 interfaces/       # User-facing adapters (CLI, API)
```

##  Quick Start

### 1. Environment Setup

```powershell
# Activate virtual environment
.\chatbot-env\Scripts\Activate.ps1

# Or use helper script
.\activate_env.ps1
```

### 2. Configuration

```powershell
# Copy environment template
Copy-Item .env.example .env

# Edit .env with your settings
notepad .env
```

### 3. Download NLTK Data

```powershell
python scripts/setup_nltk_data.py
```

### 4. Run Tests

```powershell
# All tests
pytest

# Unit tests only
pytest tests/unit -v

# With coverage
pytest --cov=src --cov-report=html
```

##  Project Structure

```
chatbot-project/

 config/                      # Configuration (12-Factor)
    settings.py             # Environment-based settings
    logging.yml             # Logging configuration

 src/                        # Application source
    domain/                 # Business Logic Layer
       entities/          # Domain entities
       value_objects/     # Immutable value objects
       services/          # Domain services
           text_preprocessor.py  # ← Preprocessing logic
   
    application/            # Application Layer
       use_cases/         # Business workflows
       dto/               # Data transfer objects
   
    infrastructure/         # Infrastructure Layer
       ml/                # ML implementations
          models/       # AIML, DialoGPT, Transformers
          embeddings/   # Word2Vec, BERT
          training/     # Training logic
       repositories/      # Data access
       external/          # External APIs
   
    interfaces/             # Interface Layer
        cli/               # Command-line interface
        api/               # REST API

 tests/                      # All tests (separate!)
    unit/                  # Unit tests
       domain/
           test_text_preprocessor.py  # ← Tests here!
    integration/           # Integration tests
    e2e/                   # End-to-end tests

 notebooks/                  # Research notebooks
 data/                       # Data storage (gitignored)
 models/                     # Trained models (gitignored)
 logs/                       # Application logs (gitignored)
 scripts/                    # Utility scripts
 docs/                       # Documentation
```

##  Key Principles

### 1. Domain-Driven Design (DDD)

**Domain Layer** (Pure Business Logic):
```python
# src/domain/services/text_preprocessor.py
# No I/O, no frameworks, just logic
class TextPreprocessingService:
    def preprocess(self, text: str) -> str:
        # Pure transformation
        ...
```

**Application Layer** (Use Cases):
```python
# src/application/use_cases/process_message.py
# Orchestrates domain services
class ProcessMessageUseCase:
    def execute(self, message_dto):
        # Coordinates domain logic
        ...
```

**Infrastructure Layer** (Implementation):
```python
# src/infrastructure/ml/models/dialogpt_chatbot.py
# Concrete implementations
class DialogGPTChatbot:
    # HuggingFace integration
    ...
```

### 2. 12-Factor App Compliance

**I. Codebase**: One codebase tracked in Git  
**II. Dependencies**: Explicit in requirements.txt  
**III. Config**: Environment variables (.env)  
**IV. Backing Services**: Attachable resources  
**V. Build, Release, Run**: Strict separation  
**VI. Processes**: Stateless execution  
**VII. Port Binding**: Self-contained services  
**VIII. Concurrency**: Scale via processes  
**IX. Disposability**: Fast startup/shutdown  
**X. Dev/Prod Parity**: Keep environments similar  
**XI. Logs**: Event streams to stdout  
**XII. Admin Processes**: One-off admin tasks  

### 3. Test Separation

Tests are **completely separate** from production code:

```
src/domain/services/text_preprocessor.py  # Production code
tests/unit/domain/test_text_preprocessor.py  # Tests
```

##  Configuration

All configuration via environment variables:

```bash
# .env
APP_ENV=development
LOG_LEVEL=INFO
BATCH_SIZE=32
LEARNING_RATE=5e-5
...
```

Access in code:

```python
from config import settings

batch_size = settings.batch_size  # Type-safe!
preprocessing_config = settings.get_preprocessing_config()
```

##  Testing

### Run Tests

```powershell
# All tests
pytest

# Specific test file
pytest tests/unit/domain/test_text_preprocessor.py

# By marker
pytest -m unit              # Unit tests only
pytest -m "not slow"        # Skip slow tests

# With coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html     # View coverage report
```

### Test Structure

- **Unit Tests**: Fast, isolated, mock all dependencies
- **Integration Tests**: Test component interactions
- **E2E Tests**: Full system tests

##  Usage Examples

### Preprocessing Text

```python
from src.domain.services.text_preprocessor import TextPreprocessingService

# Create service
preprocessor = TextPreprocessingService()

# Preprocess text
text = "I'm feeling anxious! Visit https://example.com"
clean = preprocessor.preprocess(text)
# Result: "feeling anxious visit"

# Compare methods
results = preprocessor.compare_preprocessing_methods(text)
# {'original': '...', 'stemmed': '...', 'lemmatized': '...'}
```

### Using Configuration

```python
from config import settings

# Get preprocessing config
config = settings.get_preprocessing_config()
# {'lowercase': True, 'remove_stopwords': True, ...}

# Use in service
preprocessor = TextPreprocessingService(
    language=config['language']
)
```

##  Dependencies

Core packages:
- **nltk**: Text preprocessing
- **spacy**: Advanced NLP
- **transformers**: Pre-trained models
- **torch**: Deep learning
- **pydantic-settings**: Type-safe configuration
- **pytest**: Testing framework

See [requirements.txt](requirements.txt) for full list.

##  Migration from Old Structure

The old `src/data/preprocessor.py` has been migrated:

- **Code**  `src/domain/services/text_preprocessor.py`
- **Tests**  `tests/unit/domain/test_text_preprocessor.py`
- **Config**  `.env` and `config/settings.py`

##  Development Workflow

### 1. Add New Feature

```powershell
# 1. Create domain logic
src/domain/services/new_feature.py

# 2. Create tests
tests/unit/domain/test_new_feature.py

# 3. Run tests
pytest tests/unit/domain/test_new_feature.py

# 4. Create use case (if needed)
src/application/use_cases/use_new_feature.py

# 5. Add infrastructure (if needed)
src/infrastructure/...
```

### 2. Add Configuration

```bash
# 1. Add to .env.example
NEW_CONFIG_VALUE=default

# 2. Add to config/settings.py
class Settings(BaseSettings):
    new_config_value: str = "default"

# 3. Use in code
from config import settings
value = settings.new_config_value
```

##  Deployment

The 12-Factor structure makes deployment easy:

1. **Build**: `pip install -r requirements.txt`
2. **Configure**: Set environment variables
3. **Run**: `python -m src.interfaces.cli`

##  Documentation

- [Architecture](PROJECT_STRUCTURE.md) - Detailed architecture
- [12-Factor Principles](PROJECT_STRUCTURE.md#12-factor-app-implementation)
- [Testing Guide](pytest.ini) - Test configuration

##  Contributing

1. Write tests first (TDD)
2. Keep domain logic pure
3. Use dependency injection
4. Follow layer boundaries
5. Update documentation

##  License

Academic project for NLP course.

---

**Project Status**: Production-Ready Architecture  
**Last Updated**: January 5, 2026
