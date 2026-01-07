# Quick Reference Card - Chatbot Environment

## Activate Environment
```powershell
.\chatbot-env\Scripts\Activate.ps1
# or
.\activate_env.ps1
```

## Verify Setup
```powershell
python test_environment.py
```

## Common Tasks

### Start Jupyter
```powershell
jupyter notebook
```

### Load Datasets
```python
from datasets import load_dataset
therapy = load_dataset("Amod/mental_health_counseling_conversations")
dialogs = load_dataset("daily_dialog")
```

### Test AIML Chatbot
```python
import aiml
kernel = aiml.Kernel()
# Load your AIML files here
```

### Test Transformers
```python
from transformers import pipeline
chatbot = pipeline("conversational", model="microsoft/DialoGPT-small")
```

### Test spaCy
```python
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("Your text here")
```

## Installed Package Versions
- Python: 3.11.9
- PyTorch: 2.9.1
- Transformers: 4.57.3
- spaCy: 3.8.11
- NLTK: 3.9.2
- scikit-learn: 1.8.0

## Files Created
1. `SETUP_NOTES.md` - Detailed documentation
2. `ENVIRONMENT_READY.md` - Quick start guide
3. `test_environment.py` - Verification script
4. `activate_env.ps1` - Easy activation
5. `QUICK_REFERENCE.md` - This file

## Need Help?
- Check `SETUP_NOTES.md` for troubleshooting
- Run `python test_environment.py` to diagnose issues
- Ensure you're in the activated environment (see `(chatbot-env)` in prompt)
