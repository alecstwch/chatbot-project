# Day 4 Quick Reference - Transformer Components

Quick reference for using the transformer-based components created in Day 4.

---

##  Quick Start Examples

### 1. Intent Classification

```python
from src.domain.services.intent_classifier import IntentClassificationService

# Initialize
classifier = IntentClassificationService()
classifier.load_model()

# Classify a single text
result = classifier.classify("I'm feeling very anxious")
print(f"Intent: {result.intent}")
print(f"Confidence: {result.confidence}")
print(f"All scores: {result.all_scores}")

# Batch classification
texts = ["I feel sad", "Hello there", "I'm stressed"]
results = classifier.batch_classify(texts)
for text, result in zip(texts, results):
    print(f"{text} → {result.intent}")

# Get keywords for an intent
keywords = classifier.get_intent_keywords('anxiety')
print(keywords)  # ['anxious', 'worried', 'nervous', ...]
```

### 2. Response Generation (GPT-2)

```python
from src.infrastructure.ml.models.response_generator import (
    ResponseGenerationService,
    GenerationConfig
)

# Initialize
generator = ResponseGenerationService(model_name="gpt2")
generator.load_model()

# Generate basic response
prompt = "User: I need help\nAssistant:"
response = generator.generate_response(prompt)
print(response)

# Generate therapy-focused response
response = generator.generate_therapy_response(
    user_input="I'm feeling anxious",
    intent="anxiety"
)
print(response)

# Custom generation parameters
config = GenerationConfig(
    max_length=80,
    temperature=0.6,
    top_p=0.85
)
response = generator.generate_response(prompt, config)

# Generate multiple candidates
responses = generator.generate_multiple_responses(
    prompt="Patient: I'm sad\nTherapist:",
    num_responses=3
)
for i, resp in enumerate(responses, 1):
    print(f"Option {i}: {resp}")
```

### 3. Hybrid Chatbot (AIML + GPT-2 + Intent)

```python
from src.infrastructure.ml.chatbots.hybrid_chatbot import HybridChatbot
from pathlib import Path

# Initialize
bot = HybridChatbot(
    aiml_dir=Path("data/knowledge_bases/aiml"),
    gpt2_model="gpt2",
    use_intent_classification=True,
    aiml_confidence_threshold=10
)

# Load all components
bot.initialize()

# Simple response
response = bot.respond("I feel anxious")
print(response)

# Response with metadata
result = bot.respond("I feel anxious", return_metadata=True)
print(f"Response: {result['response']}")
print(f"Strategy: {result['metadata']['strategy']}")  # 'aiml' or 'gpt2'
print(f"Intent: {result['metadata']['intent']}")
print(f"Confidence: {result['metadata']['confidence']}")

# Interactive chat
bot.chat()

# Get usage statistics
stats = bot.get_statistics()
print(f"Total: {stats['total_queries']}")
print(f"AIML: {stats['aiml_responses']} ({stats['aiml_percentage']:.1f}%)")
print(f"GPT-2: {stats['gpt2_responses']} ({stats['gpt2_percentage']:.1f}%)")

# Reset stats
bot.reset_statistics()
```

### 4. Transformer-Enhanced DialoGPT

```python
from src.infrastructure.ml.chatbots.transformer_enhanced_chatbot import TransformerEnhancedChatbot

# Initialize
bot = TransformerEnhancedChatbot(use_intent_classification=True)
bot.load_models()

# Simple response
response = bot.respond("Hi, I'm feeling anxious")
print(response)

# Response with metadata
result = bot.respond("I need help", return_metadata=True)
print(f"Response: {result['response']}")
print(f"Intent: {result['metadata']['intent']}")
print(f"Confidence: {result['metadata']['confidence']}")
print(f"Tokens: {result['metadata']['tokens_generated']}")

# Interactive chat
bot.chat()

# Get conversation history
history = bot.get_conversation_history()
for turn in history:
    print(f"User: {turn['user']}")
    print(f"Bot: {turn['bot']}")
    print(f"Intent: {turn['intent']} ({turn['confidence']:.2f})")
    print()

# Reset conversation
bot.reset_conversation()
```

---

## 🎮 Running CLI Interfaces

### AIML Chatbot (Traditional)
```bash
python -m src.interfaces.cli.chatbot_cli
```

### DialoGPT Chatbot (Neural)
```bash
python -m src.interfaces.cli.neural_chatbot_cli
```

### Hybrid Chatbot (AIML + GPT-2 + Intent)
```bash
python -m src.interfaces.cli.hybrid_chatbot_cli
```

### Transformer-Enhanced (DialoGPT + Intent)
```bash
python -m src.interfaces.cli.transformer_chatbot_cli
```

### Comprehensive Demo (All Models)
```bash
python scripts/day4_demo.py
```

---

## 🧪 Running Tests

### Test Intent Classifier
```bash
pytest tests/unit/domain/test_intent_classifier.py -v
```

### Test Response Generator
```bash
pytest tests/unit/infrastructure/test_response_generator.py -v
```

### Run All Day 4 Tests
```bash
pytest tests/unit/domain/test_intent_classifier.py tests/unit/infrastructure/test_response_generator.py -v
```

### With Coverage
```bash
pytest tests/unit/domain/test_intent_classifier.py \
  --cov=src/domain/services/intent_classifier \
  --cov-report=html

pytest tests/unit/infrastructure/test_response_generator.py \
  --cov=src/infrastructure/ml/models/response_generator \
  --cov-report=html
```

---

##  Configuration Options

### Intent Classification

**Model Options:**
- `facebook/bart-large-mnli` (default, best accuracy)
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (faster)
- `valhalla/distilbart-mnli-12-1` (smallest)

**Parameters:**
- `device`: 'cpu', 'cuda', or None (auto-detect)
- `use_keyword_fallback`: True/False (fallback to keywords)

### Response Generation

**Model Options:**
- `gpt2` (default, 124M parameters)
- `gpt2-medium` (355M, better quality)
- `gpt2-large` (774M, best but slow)
- `gpt2-xl` (1.5B, research only)

**Generation Parameters:**
- `temperature`: 0.5-1.0 (0.6-0.7 for therapy)
- `top_p`: 0.8-0.95 (nucleus sampling)
- `top_k`: 30-50 (top-k sampling)
- `repetition_penalty`: 1.1-1.3
- `max_length`: 50-150 tokens
- `min_length`: 10-30 tokens

### Hybrid Chatbot

**Parameters:**
- `aiml_confidence_threshold`: Minimum chars in AIML response (default: 10)
- `use_intent_classification`: Enable/disable intent detection
- `gpt2_model`: GPT-2 variant to use
- `intent_model`: Intent classifier model

### Transformer-Enhanced

**Parameters:**
- All DialoGPT settings from `DialoGPTSettings`
- `use_intent_classification`: Enable/disable intent detection
- Adaptive generation based on detected intent

---

##  Therapy Intents Reference

### Supported Intents

| Intent | Keywords | When to Use |
|--------|----------|-------------|
| `depression` | depressed, sad, hopeless, worthless | Expressions of sadness, hopelessness |
| `anxiety` | anxious, worried, nervous, panic | Worry, fear, nervousness |
| `stress` | stressed, overwhelmed, pressure | Work stress, burnout |
| `grief` | grief, loss, died, mourning | Loss of loved ones |
| `relationship` | partner, spouse, marriage, conflict | Relationship issues |
| `greeting` | hello, hi, good morning | Opening conversation |
| `farewell` | bye, goodbye, thanks | Closing conversation |
| `general` | (everything else) | General questions |

### Adding Custom Intents

```python
from src.domain.services.intent_classifier import IntentClassificationService

classifier = IntentClassificationService()

# Use custom candidate labels
custom_labels = ['happy', 'sad', 'angry', 'neutral']
result = classifier.classify(
    "I'm feeling great today!",
    candidate_labels=custom_labels
)
```

---

##  Troubleshooting

### Model Loading Issues

**Problem:** Model download fails
```python
# Solution: Set cache directory
from pathlib import Path

generator = ResponseGenerationService(
    cache_dir=Path("models/cache")
)
```

**Problem:** CUDA out of memory
```python
# Solution: Force CPU usage
bot = HybridChatbot(device='cpu')
```

### Performance Issues

**Problem:** Slow initialization
```python
# Solution: Use smaller models
classifier = IntentClassificationService(
    model_name="valhalla/distilbart-mnli-12-1"  # Smaller, faster
)

generator = ResponseGenerationService(
    model_name="gpt2"  # Don't use gpt2-large unless needed
)
```

**Problem:** Slow generation
```python
# Solution: Reduce max_length
config = GenerationConfig(
    max_length=50,  # Shorter responses
    top_k=30  # Faster sampling
)
```

### Quality Issues

**Problem:** Poor response quality
```python
# Solution: Adjust temperature and penalties
config = GenerationConfig(
    temperature=0.7,  # Higher = more creative
    repetition_penalty=1.3,  # Avoid repetition
    top_p=0.9
)
```

**Problem:** Intent misclassification
```python
# Solution: Use keyword fallback
classifier = IntentClassificationService(
    use_keyword_fallback=True
)

# Or add more keywords
classifier.keyword_patterns['custom_intent'] = ['word1', 'word2']
```

---

##  Best Practices

### 1. Model Selection
- Use **AIML** for known patterns (fastest)
- Use **DialoGPT** for open conversation
- Use **Hybrid** for therapy/support (best overall)
- Use **Enhanced** for intent-aware conversation

### 2. Resource Management
- Load models once, reuse instances
- Use CPU for development, GPU for production
- Cache models to avoid re-downloading
- Clear conversation history periodically

### 3. Response Quality
- Use lower temperature (0.6-0.7) for therapy
- Use higher repetition penalty (1.2-1.3)
- Keep responses short (50-100 tokens)
- Validate intent confidence before using

### 4. Error Handling
- Always enable keyword fallback
- Implement try-catch around model calls
- Provide generic fallback responses
- Log errors for analysis

---

##  Additional Resources

- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [Zero-Shot Classification Guide](https://huggingface.co/tasks/zero-shot-classification)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [DialoGPT Paper](https://arxiv.org/abs/1911.00536)
- [Intent Classification Best Practices](https://huggingface.co/blog/zero-shot-learning-nlp)

---

## Quick Checklist

Before using Day 4 components:

- [ ] Install dependencies: `pip install transformers torch`
- [ ] Verify AIML files exist in `data/knowledge_bases/aiml/`
- [ ] Check available disk space (~5GB for all models)
- [ ] Test with small examples first
- [ ] Monitor memory usage
- [ ] Review logs for errors
- [ ] Consider CPU vs GPU based on hardware

---

For complete documentation, see [DAY4.md](DAY4.md)
