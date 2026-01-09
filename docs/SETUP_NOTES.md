# Setup Notes for Chatbot Project

## Environment Setup Completed **Date:** January 5, 2026  
**Python Version:** 3.11.9  
**Virtual Environment:** `chatbot-env`

---

## Installation Summary

### Successfully Installed Packages

#### Core NLP Libraries
- nltk 3.9.2
- spacy 3.8.11 (with en_core_web_sm model)
- transformers 4.57.3
- datasets 4.4.2
- torch 2.9.1
- tokenizers 0.22.1

#### Traditional Chatbot
- python-aiml 0.9.3
- ChatterBot - **NOT INSTALLED** (see note below)

#### Machine Learning
- scikit-learn 1.8.0
- scipy 1.16.3

#### Data Processing
- pandas 2.3.3
- numpy 2.3.5

#### Visualization
- matplotlib 3.10.8
- seaborn 0.13.2
- plotly 6.5.0

#### Explainability
- lime 0.2.0.1
- shap 0.50.0

#### Jupyter & Development
- jupyter 1.1.1
- notebook 7.5.1
- ipywidgets 8.1.8
- jupyterlab 4.5.1

#### Additional NLP Models
- sentence-transformers 5.2.0
- gensim 4.4.0

---

## NLTK Data Downloaded

The following NLTK packages have been downloaded:
- punkt (tokenization)
- stopwords (stopword removal)
- wordnet (lemmatization)
- averaged_perceptron_tagger (POS tagging)
- maxent_ne_chunker (named entity recognition)
- words (word lists)

---

## SpaCy Model Downloaded

- en_core_web_sm (English language model v3.8.0)

---

## ChatterBot Status **ChatterBot is NOT installed** due to compatibility issues with Python 3.11.

### Why ChatterBot Was Excluded

ChatterBot (1.0.8) has the following issues:
1. Last updated in 2020, no longer actively maintained
2. Incompatible with SQLAlchemy 2.x (requires SQLAlchemy 1.x)
3. Dependency conflicts with Python 3.11+
4. Uses deprecated dependencies

### Alternative Approaches for the Project

According to your project plan, you need **3 model types**:
1. **Traditional/Rule-based:** Use **python-aiml** (already installed)
2. **Neural:** Use **DialoGPT/BlenderBot** via transformers (already installed)
3. **Transformer:** Use **DistilBERT/GPT-2** via transformers (already installed)

**You don't actually need ChatterBot!** The python-aiml package covers the rule-based approach, and the transformers library covers both neural and transformer approaches.

### If You Really Need ChatterBot

If the project requirements specifically mandate ChatterBot, you would need to:

**Option 1:** Use Python 3.7 or 3.8 (create separate environment)
```powershell
# Install Python 3.8 first via py launcher
py install 3.8
# Then create new environment
py -3.8 -m venv chatterbot-env
```

**Option 2:** Use a fork that's compatible with newer Python
```powershell
pip install chatterbot-corpus
pip install git+https://github.com/gunthercox/ChatterBot.git@master
```

**Option 3:** Use a modern alternative
- Rasa (more complex but production-ready)
- BotPress (open-source)
- Custom implementation using transformers

---

## Activation Instructions

### Windows PowerShell
```powershell
cd c:\Users\Alecs\chatbot-project
.\chatbot-env\Scripts\Activate.ps1
```

### Windows Command Prompt
```cmd
cd c:\Users\Alecs\chatbot-project
chatbot-env\Scripts\activate.bat
```

---

## Quick Test

To verify everything works:

```python
import nltk
import spacy
import transformers
from transformers import pipeline
import torch

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Test transformers
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")
print(result)

# Test NLTK
from nltk.tokenize import word_tokenize
tokens = word_tokenize("Hello, how are you?")
print(tokens)
```

---

## Next Steps

As per your QUICKSTART_7DAY.md plan:

### Day 1 Tasks Completed - [x] Create environment (Python 3.11)
- [x] Install all libraries
- [x] Download NLTK data
- [x] Download spaCy model

### Ready to Start Day 1 Work
1. Load datasets from HuggingFace:
   ```python
   from datasets import load_dataset
   therapy_data = load_dataset("Amod/mental_health_counseling_conversations")
   dialog_data = load_dataset("daily_dialog")
   ```

2. Quick EDA and visualizations

3. Data preprocessing pipeline

---

## Troubleshooting

### If packages fail to import
```powershell
.\chatbot-env\Scripts\Activate.ps1
pip list  # Check installed packages
```

### If NLTK data is missing
```python
import nltk
nltk.download('all')  # Download everything (may take time)
```

### If spaCy model is missing
```powershell
python -m spacy download en_core_web_sm
```

---

## Environment Info

**Virtual Environment Location:** `c:\Users\Alecs\chatbot-project\chatbot-env`

**Python Executable:** `c:\Users\Alecs\chatbot-project\chatbot-env\Scripts\python.exe`

**Pip Location:** `c:\Users\Alecs\chatbot-project\chatbot-env\Scripts\pip.exe`

---

## Package Versions Summary

```
Python: 3.11.9
PyTorch: 2.9.1
Transformers: 4.57.3
spaCy: 3.8.11
NLTK: 3.9.2
scikit-learn: 1.8.0
pandas: 2.3.3
numpy: 2.3.5
matplotlib: 3.10.8
```

---

**Setup Status:** **COMPLETE AND READY TO USE**

You can now proceed with Day 1 of your 7-day sprint! 
