# DDD Refactoring - Quick Reference

## What Changed?

### 1. Extracted Core Conversation Engine - **File:** `src/domain/services/conversation_engine.py`
- **Purpose:** Separates conversation logic (domain) from model infrastructure
- **Key Classes:**
  - `ConversationEngine` - Manages conversation flow
  - `ConversationFormatter` - Strategy pattern for different formats
  - `LanguageModelProtocol` - Interface for any language model

### 2. Created Neural Language Model (Infrastructure) - **File:** `src/infrastructure/ml/models/neural_language_model.py`
- **Purpose:** Handles model loading, tokenization, generation
- **Implements:** `LanguageModelProtocol`

### 3. Externalized All Configuration Data **Therapy Intents:**
- `config/model_configs/therapy_intents/intents.yaml` - 11 intents
- `config/model_configs/therapy_intents/keywords.yaml` - Keyword patterns

**Chef Intents:**
- `config/model_configs/chef_intents/intents.yaml` - 10 intents + funnel
- `config/model_configs/chef_intents/keywords.yaml` - Recipe keywords
- `config/model_configs/chef_intents/ingredients.yaml` - 100+ ingredients

### 4. Made Intent Classifier Domain-Agnostic - **File:** `src/domain/services/intent_classifier.py`
- **Change:** Now accepts `domain` parameter
- **Usage:**
  ```python
  therapy_classifier = IntentClassificationService(domain="therapy_intents")
  chef_classifier = IntentClassificationService(domain="chef_intents")
  ```

### 5. Created Master Chef Chatbot - **File:** `src/domain/services/chef_intent_classifier.py`
- **Features:**
  - Q&A funnel (max 5 questions)
  - Ingredient extraction
  - Constraint detection
  - Recipe recommendation

---

## Quick Test

Run this to verify everything works:
```bash
python tests/test_ddd_refactoring.py
```

Expected output: ALL TESTS PASSED ---

## Usage Examples

### Intent Classification (Therapy)
```python
from src.domain.services.intent_classifier import IntentClassificationService

# Initialize for therapy domain
classifier = IntentClassificationService(domain="therapy_intents")

# No model needed for keyword fallback
result = classifier._keyword_classify("I feel so depressed")
print(f"Intent: {result.intent}, Confidence: {result.confidence}")
```

### Intent Classification (Chef)
```python
# Initialize for chef domain
chef_classifier = IntentClassificationService(domain="chef_intents")

# Get domain info
info = chef_classifier.get_domain_info()
print(f"Loaded {info['num_intents']} chef intents")
```

### Conversation Engine
```python
from src.domain.services.conversation_engine import (
    ConversationEngine,
    SimpleConversationFormatter
)

# Mock model for testing
class MockModel:
    def generate(self, prompt, **kwargs):
        return prompt + "AI response here"
    def is_ready(self):
        return True

engine = ConversationEngine(
    model=MockModel(),
    formatter=SimpleConversationFormatter()
)

response = engine.generate_response("Hello")
```

### Chef Chatbot
```python
from src.domain.services.chef_intent_classifier import ChefIntentClassifier

chef = ChefIntentClassifier(max_funnel_questions=5)
# Note: Loads ML model
# chef.load_model()

# For now, test without model
context = chef.get_context()
print(f"Funnel complete: {context.is_complete()}")
```

---

## Configuration Structure

```
config/model_configs/
 therapy_intents/
    intents.yaml          # Intent definitions
      - 11 therapy intents
      - Crisis detection
      - Response templates
   
    keywords.yaml          # Keyword patterns
       - Primary keywords
       - Secondary keywords
       - Phrases

 chef_intents/
     intents.yaml           # Chef intent definitions
       - 10 chef intents
       - 5 funnel stages
       - 8 dish types
       - 13 cuisines
    
     keywords.yaml          # Recipe keywords
       - Ingredient keywords
       - Constraint keywords
       - Number words
    
     ingredients.yaml       # Ingredient database
        - 8 categories
        - 100+ ingredients
        - Allergen info
        - Substitutes
```

---

## Architecture Benefits

### DDD (Domain-Driven Design)
- **Clear Boundaries** - Domain logic separate from infrastructure
- **Ubiquitous Language** - ConversationTurn, IntentPrediction, RecipeContext
- **Layered Architecture** - Domain  Infrastructure
- **Dependency Inversion** - Depends on protocols, not implementations

### 12-Factor App
- **Config** - All configuration externalized (YAML files)
- **Dependencies** - Explicitly declared (requirements.txt, pyyaml added)
- **Backing Services** - Models treated as attached resources
- **Dev/Prod Parity** - Same code, different configs

### Testing
- **Easy to Mock** - LanguageModelProtocol can be mocked
- **No Model Required** - Keyword classification works without ML
- **Configuration Testing** - Test YAML loading separately
- **Unit vs Integration** - Clear separation

---

## Files Created

1. **Domain Layer:**
   - `src/domain/services/conversation_engine.py` (298 lines)
   - `src/domain/services/chef_intent_classifier.py` (366 lines)
   
2. **Infrastructure Layer:**
   - `src/infrastructure/ml/models/neural_language_model.py` (211 lines)
   - `src/infrastructure/ml/chatbots/neural_chatbot_ddd.py` (179 lines)
   
3. **Configuration:**
   - `config/model_configs/therapy_intents/intents.yaml` (119 lines)
   - `config/model_configs/therapy_intents/keywords.yaml` (201 lines)
   - `config/model_configs/chef_intents/intents.yaml` (186 lines)
   - `config/model_configs/chef_intents/keywords.yaml` (174 lines)
   - `config/model_configs/chef_intents/ingredients.yaml` (232 lines)
   
4. **Tests & Demos:**
   - `tests/test_ddd_refactoring.py` (120 lines)
   - `scripts/demo_chef_chatbot.py` (95 lines)
   
5. **Documentation:**
   - `docs/DDD_REFACTORING.md` (comprehensive guide)
   - `docs/DDD_QUICK_REFERENCE.md` (this file)

---

## Next Steps

1. **Test Chef Demo:**
   ```bash
   # This will load the transformers model
   python scripts/demo_chef_chatbot.py
   ```

2. **Integrate with Existing Chatbots:**
   - Update existing chatbots to use new architecture
   - Create CLI interfaces for chef chatbot

3. **Add More Domains:**
   - Create `financial_intents` for financial chatbot
   - Create `travel_intents` for travel chatbot
   - Simply add new YAML configs!

4. **Production Deployment:**
   - All configs in version control
   - Easy to A/B test different keyword sets
   - Non-technical users can update keywords

---

## Troubleshooting

### Import Errors
If you get "No module named 'yaml'":
```bash
pip install pyyaml==6.0.2
```

### Configuration Not Found
Check that YAML files exist:
```bash
ls config/model_configs/therapy_intents/
ls config/model_configs/chef_intents/
```

### Syntax Errors
The intent_classifier.py was recreated from scratch if corrupted.
If issues persist, check line endings (should be LF, not CRLF on Windows).

---

## Summary

**Status:** ALL REFACTORING COMPLETE

**Achievement:**
- Extracted core engines following DDD
- Externalized ALL hardcoded data to YAML  
- Created extensible, domain-agnostic architecture
- Added Master Chef Q&A funnel system
- All tests passing

**Impact:**
- Easy to add new chatbot domains (just add YAML configs)
- Easy to swap ML models (implements protocol)
- Easy to test (can mock components)
- Production-ready architecture (12-Factor compliant)
