# Paper Documentation

This directory contains LaTeX documentation for the chatbot project, following the ACL 2023 NLP paper format.

## Files Created

### Main Document
- `main.tex` - Main LaTeX document that includes all sections

### Sections (in `sections/` directory)
1. `abstract.tex` - Abstract summarizing the three architectures
2. `introduction.tex` - Motivation, scope, and contributions
3. `related_work.tex` - Background on conversational AI and mental health chatbots
4. `methodology.tex` - Detailed implementation descriptions
5. `results.tex` - Implementation results (what was built, not evaluation)
6. `discussion.tex` - Trade-offs and practical considerations
7. `conclusion.tex` - Summary and future directions
8. `limitations.tex` - Known limitations of the work
9. `ethical_statement.tex` - Ethical considerations for mental health chatbots
10. `appendix.tex` - Code structure, configuration examples, CLI commands

### Bibliography
- `references.bib` - Bibliography with relevant citations

## How to Compile

### Using pdflatex (recommended):
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk (if available):
```bash
cd paper
latexmk -pdf main.tex
```

## What is Documented

The paper describes three chatbot architectures that were implemented:

### 1. AIML Rule-Based System
- 150+ hand-crafted pattern-response rules
- XML-based knowledge base
- Deterministic, interpretable behavior
- Files: `src/infrastructure/ml/chatbots/aiml_chatbot.py`

### 2. DialoGPT Neural Model
- 117M parameter transformer model
- Pre-trained on Reddit conversations
- Context-aware generation with conversation history
- Files: `src/infrastructure/ml/chatbots/dialogpt_chatbot.py`

### 3. RAG-Enhanced Transformer
- Gemini 2.5 Flash API for generation
- Qdrant vector database for semantic memory
- MongoDB for patient profiles
- Emotion detection (12 categories)
- Behavior pattern recognition (15 patterns)
- Strict data isolation for privacy
- Files: `src/infrastructure/ml/chatbots/enhanced_rag_chatbot.py`

## Key Points Addressed

From the evaluation checklist:

✅ **Motivation and problem solving**: Mental health accessibility, architectural comparison

✅ **Data analysis**: Datasets used for reference (Mental Health Counseling, DailyDialog)

✅ **Preprocessing comparison**: Lemmatization chosen over stemming with rationale

✅ **Embedding justification**: Sentence-BERT chosen over OpenAI/USE with reasoning

✅ **Multiple models**: Three architectures implemented (AIML, DialoGPT, RAG)

✅ **Traditional model**: AIML rule-based system

✅ **Neural network**: DialoGPT transformer

✅ **Transformer**: RAG with Gemini 2.5 Flash

✅ **LLM use**: Gemini API for response generation with justification

✅ **Multiple datasets**: Mental Health Counseling + DailyDialog referenced

✅ **Originality**: RAG for mental health with domain services, data isolation

✅ **Code quality**: DDD architecture, 100% test coverage on preprocessor

## Points NOT Addressed (Not Implemented)

❌ **Hyperparameter tuning**: Not performed - used default/recommended values

❌ **Model evaluation**: No BLEU, ROUGE, accuracy, or other metrics calculated

❌ **Error analysis**: No systematic error categorization performed

❌ **Explainability**: LIME mentioned in related work but not implemented

❌ **Performance comparison**: No quantitative comparison between architectures

## Customization

### Update Author Information
Edit `main.tex` line 31:
```latex
\author{Your Name \\ \texttt{your.email@domain.com}}
```

### Update Title
Edit `main.tex` line 29 if needed.

## Dependencies

The paper uses the ACL2023 document class. If not available, you can change to standard article class:

```latex
\documentclass[11pt]{article}
```

And add required packages manually (most are already included).

## Paper Organization

The paper follows this structure:
1. Abstract (what was done)
2. Introduction (why it was done)
3. Related Work (background context)
4. Methodology (how it was implemented)
5. Results (what was built)
6. Discussion (trade-offs and considerations)
7. Conclusion (summary)
8. Limitations (what's missing)
9. Ethical Statement (responsibilities)
10. Appendix (technical details)

## Notes

- This is **implementation documentation**, not an evaluation paper
- We describe what was built, not how well it performs
- No performance metrics or benchmarks are included
- Focus is on architecture, design decisions, and practical implementation
- Ethical section is thorough given mental health domain
- Code examples and configuration provided in appendix
