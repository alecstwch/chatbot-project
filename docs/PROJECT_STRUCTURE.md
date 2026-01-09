# Chatbot Project Structure

## Architecture Principles

### 1. 12-Factor App Compliance
- **Codebase**: One codebase in Git, many deploys
- **Dependencies**: Explicit declaration (requirements.txt)
- **Config**: Store config in environment variables (.env)
- **Backing Services**: Treat as attached resources
- **Build, Release, Run**: Strict separation
- **Processes**: Execute as stateless processes
- **Port Binding**: Export services via port binding
- **Concurrency**: Scale out via process model
- **Disposability**: Fast startup, graceful shutdown
- **Dev/Prod Parity**: Keep environments similar
- **Logs**: Treat logs as event streams
- **Admin Processes**: Run as one-off processes

### 2. Domain-Driven Design (DDD)
- **Domain Layer**: Core business logic (chatbot entities, NLP logic)
- **Application Layer**: Use cases and orchestration
- **Infrastructure Layer**: External concerns (databases, APIs, file I/O)
- **Interface Layer**: User-facing components (CLI, API, notebooks)

### 3. Test Separation
- **Unit Tests**: Separate from production code
- **Integration Tests**: Test component interactions
- **E2E Tests**: Full workflow testing

---

## Project Structure

```
chatbot-project/
│
├── .env.example                    # Example environment variables
├── .gitignore                      # Git ignore rules
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
├── pyproject.toml                  # Modern Python project config
│
├── config/                         # Configuration files (12-factor)
│   ├── __init__.py
│   ├── settings.py                 # Load from environment
│   ├── logging.yml                 # Logging configuration
│   └── model_configs/              # Model hyperparameters
│       ├── aiml_config.yml
│       ├── dialogpt_config.yml
│       └── transformer_config.yml
│
├── src/                            # Application source code
│   ├── __init__.py
│   │
│   ├── domain/                     # Domain Layer (Business Logic)
│   │   ├── __init__.py
│   │   ├── entities/               # Core entities
│   │   │   ├── __init__.py
│   │   │   ├── conversation.py    # Conversation entity
│   │   │   ├── message.py         # Message entity
│   │   │   └── user_profile.py    # User entity
│   │   ├── value_objects/          # Immutable values
│   │   │   ├── __init__.py
│   │   │   ├── intent.py          # Intent classification result
│   │   │   └── confidence.py      # Confidence score
│   │   └── services/               # Domain services
│   │       ├── __init__.py
│   │       ├── text_preprocessor.py   # Pure preprocessing logic
│   │       └── intent_classifier.py   # Intent classification
│   │
│   ├── application/                # Application Layer (Use Cases)
│   │   ├── __init__.py
│   │   ├── use_cases/              # Business workflows
│   │   │   ├── __init__.py
│   │   │   ├── process_user_message.py
│   │   │   ├── train_chatbot.py
│   │   │   └── evaluate_model.py
│   │   └── dto/                    # Data Transfer Objects
│   │       ├── __init__.py
│   │       ├── message_dto.py
│   │       └── response_dto.py
│   │
│   ├── infrastructure/             # Infrastructure Layer
│   │   ├── __init__.py
│   │   ├── repositories/           # Data access
│   │   │   ├── __init__.py
│   │   │   ├── conversation_repository.py
│   │   │   └── dataset_repository.py
│   │   ├── ml/                     # ML implementations
│   │   │   ├── __init__.py
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── aiml_chatbot.py
│   │   │   │   ├── dialogpt_chatbot.py
│   │   │   │   └── transformer_chatbot.py
│   │   │   ├── embeddings/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── word2vec_embedder.py
│   │   │   │   └── bert_embedder.py
│   │   │   └── training/
│   │   │       ├── __init__.py
│   │   │       ├── trainer.py
│   │   │       └── evaluator.py
│   │   ├── external/               # External services
│   │   │   ├── __init__.py
│   │   │   ├── huggingface_client.py
│   │   │   └── openai_client.py
│   │   └── persistence/            # Data storage
│   │       ├── __init__.py
│   │       ├── file_storage.py
│   │       └── cache_manager.py
│   │
│   └── interfaces/                 # Interface Layer (Adapters)
│       ├── __init__.py
│       ├── cli/                    # Command-line interface
│       │   ├── __init__.py
│       │   └── chatbot_cli.py
│       ├── api/                    # REST API (if needed)
│       │   ├── __init__.py
│       │   └── routes.py
│       └── notebooks/              # Jupyter notebooks (exploratory)
│           └── README.md
│
├── tests/                          # All tests separate from src
│   ├── __init__.py
│   ├── conftest.py                 # Pytest configuration
│   │
│   ├── unit/                       # Unit tests
│   │   ├── __init__.py
│   │   ├── domain/
│   │   │   ├── test_text_preprocessor.py
│   │   │   └── test_intent_classifier.py
│   │   ├── application/
│   │   │   └── test_use_cases.py
│   │   └── infrastructure/
│   │       └── test_models.py
│   │
│   ├── integration/                # Integration tests
│   │   ├── __init__.py
│   │   └── test_chatbot_pipeline.py
│   │
│   └── e2e/                        # End-to-end tests
│       ├── __init__.py
│       └── test_full_conversation.py
│
├── notebooks/                      # Research notebooks (12-factor: dev only)
│   ├── 01_Data_Exploration.ipynb
│   ├── 02_Preprocessing_Experiments.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Evaluation_Analysis.ipynb
│
├── data/                           # Data directory (12-factor: external)
│   ├── raw/                        # Original datasets
│   ├── processed/                  # Cleaned data
│   ├── embeddings/                 # Pre-computed embeddings
│   └── .gitkeep
│
├── models/                         # Saved models (12-factor: artifacts)
│   ├── aiml/
│   ├── dialogpt/
│   ├── transformer/
│   └── .gitkeep
│
├── logs/                           # Application logs (12-factor: streams)
│   └── .gitkeep
│
├── scripts/                        # Utility scripts
│   ├── download_datasets.py
│   ├── train_all_models.py
│   └── setup_nltk_data.py
│
├── docs/                           # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   └── deployment.md
│
└── paper/                          # LaTeX paper (academic deliverable)
    ├── main.tex
    ├── sections/
    ├── figures/
    ├── references.bib
    └── acl2023.sty
```

---

## Key Design Decisions

### Domain Layer (Core Business Logic)
```python
# src/domain/services/text_preprocessor.py
# Pure business logic, no external dependencies

class TextPreprocessingService:
    """Domain service for text preprocessing"""
    def clean_text(self, text: str) -> str:
        # Pure logic, no I/O
        pass
```

### Application Layer (Use Cases)
```python
# src/application/use_cases/process_user_message.py
# Orchestrates domain services

class ProcessUserMessageUseCase:
    """Use case: Process incoming user message"""
    def __init__(self, preprocessor, chatbot, repository):
        self.preprocessor = preprocessor
        self.chatbot = chatbot
        self.repository = repository
    
    def execute(self, message_dto: MessageDTO) -> ResponseDTO:
        # Orchestrate domain services
        pass
```

### Infrastructure Layer (Implementation Details)
```python
# src/infrastructure/ml/models/dialogpt_chatbot.py
# Concrete implementation

class DialogGPTChatbot:
    """DialoGPT implementation"""
    def __init__(self, config):
        # Load model from HuggingFace
        pass
```

### Interface Layer (User-Facing)
```python
# src/interfaces/cli/chatbot_cli.py
# CLI adapter

class ChatbotCLI:
    """Command-line interface for chatbot"""
    def __init__(self, use_case):
        self.use_case = use_case
    
    def run(self):
        # Handle user input/output
        pass
```

---

## 12-Factor App Implementation

### Configuration (.env)
```bash
# .env.example
# App Config
APP_ENV=development
LOG_LEVEL=INFO

# Model Paths
AIML_MODEL_PATH=models/aiml
DIALOGPT_MODEL_PATH=models/dialogpt

# HuggingFace
HF_CACHE_DIR=models/cache
HF_TOKEN=your_token_here

# Training
BATCH_SIZE=32
LEARNING_RATE=5e-5
MAX_EPOCHS=3
```

### Config Loader
```python
# config/settings.py
from pathlib import Path
from pydantic import BaseSettings

class Settings(BaseSettings):
    app_env: str = "development"
    log_level: str = "INFO"
    aiml_model_path: Path
    dialogpt_model_path: Path
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

---

## Testing Strategy

### Unit Tests (tests/unit/)
- Test individual functions and classes
- Mock all external dependencies
- Fast execution (< 1s each)

### Integration Tests (tests/integration/)
- Test component interactions
- Use real but lightweight dependencies
- Moderate execution time

### E2E Tests (tests/e2e/)
- Test full user workflows
- Use production-like setup
- Slower execution

---

## Benefits of This Structure

**Separation of Concerns**: Each layer has clear responsibility  
**Testability**: Easy to mock and test in isolation  
**Maintainability**: Changes isolated to specific layers  
**Scalability**: Easy to add new models or features  
**12-Factor Compliant**: Ready for deployment  
**DDD Principles**: Business logic protected in domain layer  
**Clean Tests**: Separate from production code  

---

## Migration from Current Structure

Current files will be reorganized:
- `src/data/preprocessor.py` → `src/domain/services/text_preprocessor.py` (logic only)
- Tests from `preprocessor.py` → `tests/unit/domain/test_text_preprocessor.py`
- Notebooks stay in `notebooks/` for research

---

## Next Steps

1. Create directory structure
2. Move existing code to appropriate layers
3. Extract tests to tests/ directory
4. Create .env.example
5. Update imports and paths
6. Run tests to verify migration

Would you like me to proceed with the restructuring?
