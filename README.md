# 7-Day Chatbot Project - Two Conversational Agents

## Project Overview

This repository contains the implementation of **two conversational agents** built in 7 days using state-of-the-art NLP techniques and open-source tools:

1. **Psychotherapy Chatbot** - Mental health support using CBT principles
2. **General Conversational Chatbot** - Open-domain chitchat

##  Project Goals

- Demonstrate comprehensive NLP preprocessing techniques
- Implement 3 types of models: Traditional (AIML), Neural (DialoGPT), Transformer (DistilBERT/GPT-2)
- Comparative analysis of different approaches
- Complete research paper in ACL 2023 format
- All using 100% open-source tools (zero cost)

##  Repository Structure

```
 config/                    # Configuration (12-Factor App)
    settings.py            # Type-safe settings with Pydantic
    logging.yml            # Logging configuration
 data/
    raw/                   # Original datasets
       therapy/           # Mental health counseling data
       dialogs/           # Daily dialog data
    processed/             # Preprocessed datasets
    external/              # External data sources
    knowledge_bases/       # AIML and rule-based knowledge
        aiml/              # AIML files (therapy.aiml, general.aiml)
 docs/                      # Documentation
    research_papers/       # Papers from MDs folder
    PROJECT_STRUCTURE.md   # Architecture documentation
    MIGRATION_GUIDE.md     # Restructuring guide
    RESTRUCTURING_COMPLETE.md
    QUICK_REFERENCE.md
    ENVIRONMENT_READY.md
    SETUP_NOTES.md
 notebooks/                 # Jupyter notebooks
    labs/                  # Lab notebooks from course
 src/                       # Source code (DDD layers)
    domain/                # Domain layer
       services/
           text_preprocessor.py  # Complete with tests
    application/           # Application layer
    infrastructure/        # Infrastructure layer
    interfaces/            # Interface layer (CLI/API)
 tests/                     # Test suite (separated from code)
    unit/
       domain/
           test_text_preprocessor.py  # 27 tests, 100% coverage
    integration/
    e2e/
    conftest.py
 models/                    # Trained model checkpoints
 logs/                      # Application logs
 scripts/                   # Utility scripts
    setup_nltk_data.py     # NLTK data setup
 NLP_Paper_Template/        # LaTeX paper
    main.tex
    sections/
    references.bib
 .env.example               # Environment template
 .gitignore                 # Git ignore rules
 pytest.ini                 # Pytest configuration
 requirements.txt           # Python dependencies
 setup.py                   # Package setup
 PROJECT_PLAN.md            # Detailed project plan
 QUICKSTART_7DAY.md         # Day-by-day guide
 README_NEW_STRUCTURE.md    # DDD architecture overview
 README.md                  # This file
```

##  Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n chatbot python=3.9
conda activate chatbot

# Install dependencies
pip install -r requirements.txt

# Run automated setup
python setup.py
```

### 2. Download Datasets

Datasets are automatically downloaded by `setup.py`, or manually:

```python
from datasets import load_dataset

# Psychotherapy dataset
therapy_data = load_dataset("Amod/mental_health_counseling_conversations")
therapy_data.save_to_disk("./data/raw/therapy")

# Dialog dataset
dialog_data = load_dataset("daily_dialog")
dialog_data.save_to_disk("./data/raw/dialogs")
```

### 3. Run Preprocessing

```python
from src.data.preprocessor import TextPreprocessor

preprocessor = TextPreprocessor()
processed_text = preprocessor.preprocess("Your text here")
```

### 4. Test AIML Chatbot

```python
import aiml

kernel = aiml.Kernel()
kernel.learn("aiml_files/therapy.aiml")

response = kernel.respond("I feel anxious")
print(response)
```

### 5. Run Neural Chatbot

```python
from transformers import pipeline

chatbot = pipeline("conversational", model="microsoft/DialoGPT-small")
# See notebooks for full implementation
```

##  Models Implemented

### 1. Traditional/Rule-Based (AIML)
- Pattern matching with AIML
- 50+ therapy patterns in `therapy.aiml`
- Keyword-based intent recognition

### 2. Neural Network (DialoGPT)
- Pre-trained Microsoft DialoGPT-small
- Fine-tuned on Daily Dialog dataset
- Context-aware conversation

### 3. Transformer (DistilBERT + GPT-2)
- DistilBERT for intent classification
- GPT-2 for response generation
- Hybrid approach combining rule-based + transformer

##  NLP Techniques Demonstrated

- Tokenization (NLTK word_tokenize)
- Stopword removal (comparison analysis)
- Stemming (Porter Stemmer)
- Lemmatization (WordNet Lemmatizer)
- TF-IDF vectorization
- Word embeddings (Word2Vec, GloVe)
- Contextual embeddings (BERT, DistilBERT)
- Intent classification
- Response generation
- Dialogue management

##  Evaluation Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score
- **Generation**: BLEU, ROUGE, Perplexity
- **Dialogue**: Task completion rate, response appropriateness
- **Explainability**: LIME, attention visualization

##  Documentation

- **[PROJECT_PLAN.md](PROJECT_PLAN.md)** - Complete project overview and methodology
- **[QUICKSTART_7DAY.md](QUICKSTART_7DAY.md)** - Day-by-day implementation guide
- **paper/main.tex** - Research paper (ACL 2023 format)

##  Course Requirements Met

- Data analysis with visualizations
- Multiple preprocessing methods compared
- Different embedding techniques (static + contextual)
- 3 model types: Traditional, Neural, Transformer
- Hyperparameter tuning documented
- Testing on 2 datasets (cross-domain)
- Explainability analysis
- Error analysis
- LaTeX documentation
- Complete implementation code

## Technologies Used

### Core Libraries
- **NLP**: NLTK, spaCy, transformers
- **ML**: scikit-learn, PyTorch
- **Chatbot**: python-aiml, ChatterBot
- **Data**: pandas, numpy, datasets
- **Viz**: matplotlib, seaborn, plotly

### Models
- **AIML** (rule-based)
- **DialoGPT-small** (Microsoft)
- **DistilBERT** (classification)
- **GPT-2** (generation)

### Tools
- **Compute**: Google Colab (free GPU)
- **Version Control**: Git/GitHub
- **Paper**: Overleaf (LaTeX)
- **Datasets**: HuggingFace Hub

##  Usage Examples

### Therapy Chatbot
```python
from src.models.traditional import TherapyChatbot

bot = TherapyChatbot()
response = bot.respond("I'm feeling depressed")
print(response)
# Output: "I'm sorry to hear that you're feeling depressed..."
```

### General Chatbot
```python
from src.models.neural import GeneralChatbot

bot = GeneralChatbot()
response = bot.chat("Hello! How are you?")
print(response)
# Output: Contextual response from DialoGPT
```

##  Running Tests

```bash
# Run all notebooks
jupyter notebook notebooks/

# Test preprocessing
python src/data/preprocessor.py

# Evaluate models
python src/evaluation/metrics.py
```

##  Results Summary

(To be updated after Day 5 evaluation)

| Model | BLEU | F1-Score | Response Time |
|-------|------|----------|---------------|
| AIML | TBD | TBD | TBD |
| DialoGPT | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD |

##  Contributing

This is a course project. Team members:
- [Member 1 Name]
- [Member 2 Name]
- [Member 3 Name] (if applicable)
- [Member 4 Name] (if applicable)

##  License

This project is for educational purposes. Code is provided as-is for reference.

##  Acknowledgments

- Course: Foundations of NLP
- Instructor: Zăvelcă Miruna-Andreea
- Datasets: HuggingFace community
- Models: Microsoft (DialoGPT), Google (DistilBERT)
- Papers referenced in Related Work section

##  Contact

For questions about this project, please contact [your email].

## Timeline

- **Day 1** (Jan 6): Setup & Data - **Day 2** (Jan 7): AIML Therapy Bot
- **Day 3** (Jan 8): DialoGPT Chatbot
- **Day 4** (Jan 9): Transformer Integration
- **Day 5** (Jan 10): Evaluation
- **Day 6** (Jan 11): Paper Writing
- **Day 7** (Jan 12): Finalization & Submission

---

**Status**:  In Development  
**Last Updated**: January 5, 2026  
**Version**: 1.0
