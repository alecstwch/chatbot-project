# Chatbot Configuration Guide

## Overview
This project follows the **12-Factor App** methodology by externalizing all configuration through environment variables.

## Configuration Files

### `.env`
Main configuration file containing all environment variables. Copy from `.env.example` to get started.

### `.env.chatbot.example`
Template specifically for chatbot settings.

## DialoGPT Configuration

All DialoGPT settings use the `DIALOGPT_` prefix:

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `DIALOGPT_MODEL_NAME` | `microsoft/DialoGPT-small` | - | HuggingFace model identifier |
| `DIALOGPT_MAX_HISTORY_LENGTH` | `1000` | 100-2000 | Maximum conversation history in tokens |
| `DIALOGPT_MAX_NEW_TOKENS` | `30` | 10-100 | Maximum tokens per response |
| `DIALOGPT_TEMPERATURE` | `0.7` | 0.0-2.0 | Sampling randomness (lower = more focused) |
| `DIALOGPT_TOP_P` | `0.92` | 0.0-1.0 | Nucleus sampling threshold |
| `DIALOGPT_TOP_K` | `50` | 0-100 | Top-k sampling (0 = disabled) |
| `DIALOGPT_REPETITION_PENALTY` | `1.15` | 1.0-2.0 | Penalty for repeating tokens |
| `DIALOGPT_LENGTH_PENALTY` | `1.0` | 0.0-2.0 | Length penalty for generation |
| `DIALOGPT_DEVICE` | `auto` | auto/cpu/cuda | Device to run model on |

## AIML Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AIML_DIR` | `data/knowledge_bases/aiml` | Directory containing AIML files |

## Tuning for Better Responses

### More Focused/Conservative
```env
DIALOGPT_TEMPERATURE=0.6
DIALOGPT_TOP_P=0.85
DIALOGPT_TOP_K=30
DIALOGPT_REPETITION_PENALTY=1.2
DIALOGPT_MAX_NEW_TOKENS=25
```

### More Creative/Diverse
```env
DIALOGPT_TEMPERATURE=0.9
DIALOGPT_TOP_P=0.95
DIALOGPT_TOP_K=0
DIALOGPT_REPETITION_PENALTY=1.1
DIALOGPT_MAX_NEW_TOKENS=50
```

### Balanced (Recommended)
```env
DIALOGPT_TEMPERATURE=0.7
DIALOGPT_TOP_P=0.92
DIALOGPT_TOP_K=50
DIALOGPT_REPETITION_PENALTY=1.15
DIALOGPT_MAX_NEW_TOKENS=30
```

## Usage in Code

### Python Code
```python
from src.infrastructure.config.chatbot_settings import DialoGPTSettings

# Load from environment
settings = DialoGPTSettings()

# Use in chatbot
from src.infrastructure.ml.chatbots.dialogpt_chatbot import DialoGPTChatbot
chatbot = DialoGPTChatbot(settings=settings)
```

### Command Line
```bash
# Override specific settings
DIALOGPT_TEMPERATURE=0.5 python -m src.interfaces.cli.general_chatbot_cli

# Use different model
DIALOGPT_MODEL_NAME=microsoft/DialoGPT-medium python -m src.interfaces.cli.general_chatbot_cli
```

## Troubleshooting

### Responses Too Random/Gibberish
- **Lower** `DIALOGPT_TEMPERATURE` (try 0.5-0.7)
- **Lower** `DIALOGPT_TOP_P` (try 0.85-0.90)
- **Increase** `DIALOGPT_REPETITION_PENALTY` (try 1.2-1.3)
- **Decrease** `DIALOGPT_MAX_NEW_TOKENS` (try 20-30)

### Responses Too Repetitive
- **Increase** `DIALOGPT_TEMPERATURE` (try 0.8-0.9)
- **Decrease** `DIALOGPT_REPETITION_PENALTY` (try 1.05-1.1)
- **Increase** `DIALOGPT_TOP_P` (try 0.95)

### Responses Too Short
- **Increase** `DIALOGPT_MAX_NEW_TOKENS` (try 40-60)
- **Adjust** `DIALOGPT_LENGTH_PENALTY` (try 1.2)

### Responses Too Long
- **Decrease** `DIALOGPT_MAX_NEW_TOKENS` (try 20-30)
- **Adjust** `DIALOGPT_LENGTH_PENALTY` (try 0.8)

## Best Practices

1. **Never commit `.env` to git** - It may contain sensitive data
2. **Use `.env.example` as template** - Document all required variables
3. **Override in development** - Use environment variables for testing
4. **Keep defaults sensible** - Ensure application works out-of-the-box
5. **Document changes** - Update this guide when adding new settings

## References

- [12-Factor App Config](https://12factor.net/config)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [HuggingFace Generation Parameters](https://huggingface.co/docs/transformers/main_classes/text_generation)
