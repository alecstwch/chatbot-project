# Day 6: Paper Writing IN PROGRESS

**Date:** January 7, 2026  
**Status:** IN PROGRESS  
**Completed:** 4/11 sections  
**Remaining:** 4-6 hours

---

## Overview

Day 6 focuses on writing the research paper in LaTeX (ACL 2023 format). We're documenting all experimental work from Days 1-5 in a publishable academic paper.

---

## Progress Summary

### Completed Sections (4/11)

1. **Abstract** (200 words) - Complete
   - Problem statement
   - Approach summary (3 architectures)
   - Key results (F1: 0.83, BLEU: 0.62)
   - Main finding (60% errors from intent misclassification)
   - Impact statement

2. **Introduction** (1,200 words) - Complete
   - Motivation (3 key observations)
   - Research questions (RQ1-RQ4)
   - Contributions (4 main points)
   - Paper organization

3. **Results** (1,800 words + 8 tables) - Complete
   - Table 1: Overall model comparison
   - Table 2: Intent classification metrics
   - Table 3: Response generation metrics
   - Table 4: Dialogue quality
   - Table 5: Cross-validation (5-fold)
   - Table 6: Error distribution
   - Table 7: Model complexity/efficiency
   - Key findings summary

4. **Discussion** (2,400 words) - Complete
   - Interpretation of results
   - Comparison with related work
   - Error analysis insights
   - Explainability findings (LIME)
   - Practical implications
   - Unexpected findings

###  Remaining Sections (7/11)

5. **Related Work** - NOT STARTED
   - 4 papers to review and cite
   - ~800 words
   - Estimated: 2 hours

6. **Methodology** - NOT STARTED
   - Datasets, preprocessing, architectures
   - ~1,500 words
   - Estimated: 2 hours

7. **Conclusion** - NOT STARTED
   - ~400 words
   - Estimated: 30 minutes

8. **Limitations** - NOT STARTED
   - ~300 words
   - Estimated: 20 minutes

9. **Ethical Statement** - NOT STARTED
   - ~400 words
   - Estimated: 30 minutes

10. **Acknowledgements** - NOT STARTED
    - ~100 words
    - Estimated: 10 minutes

11. **References** - NOT STARTED
    - BibTeX entries
    - Estimated: 30 minutes

---

## Files Created/Modified

### LaTeX Files

1. **`NLP_Paper_Template/main.tex`**
   - Updated title: "Comparative Analysis of Conversational Agent Architectures: Rule-Based, Neural, and Hybrid Approaches"
   - Added packages: booktabs, multirow, graphicx, amsmath, amssymb
   - Updated section imports to include results and discussion

2. **`NLP_Paper_Template/sections/abstract.tex`** (200 words)
   - Complete abstract with all key findings

3. **`NLP_Paper_Template/sections/introduction.tex`** (1,200 words)
   - Motivation section
   - Research questions
   - Contributions
   - Paper organization

4. **`NLP_Paper_Template/sections/results.tex`** (1,800 words + 8 tables)
   - Comprehensive results with LaTeX tables
   - All metrics from Day 5 evaluation

5. **`NLP_Paper_Template/sections/discussion.tex`** (2,400 words)
   - Detailed interpretation and analysis
   - Comparison with literature
   - Practical implications

---

## Key Results (From Day 5 Evaluation)

### Model Performance

| Model | Accuracy | F1 | BLEU | ROUGE-L |
|-------|----------|-----|------|---------|
| AIML | 0.72 | 0.69 | 0.45 | 0.48 |
| DialoGPT | 0.78 | 0.75 | 0.58 | 0.60 |
| **GPT-2 + Intent** | **0.85** | **0.83** | **0.62** | **0.65** |

### Error Distribution

- Intent Misclassification: 60%
- Repetitive Responses: 20%
- Generic Responses: 20%

### Cross-Validation (5-Fold)

- Accuracy: 0.825 ± 0.012
- F1-Score: 0.792 ± 0.015
- BLEU: 0.579 ± 0.023

---

## Next Steps (Remaining Work)

### Priority 1: Methodology (2 hours)

Based on actual implementation:

```latex
\section{Methodology}
\label{sec:methodology}

\subsection{Datasets}
- Mental Health Counseling Conversations (Hugging Face)
- Daily Dialog Dataset (Hugging Face)
- Statistics: [add actual numbers]
- 80/10/10 train/val/test split

\subsection{Preprocessing Pipeline}
From src/domain/services/text_preprocessor.py:
- Tokenization: NLTK word_tokenize
- Stopword removal: NLTK stopwords
- Lemmatization: WordNet lemmatizer
- URL/email removal
- Special character handling
- Unit tested: 27 tests, 100% coverage

\subsection{Model Architectures}

\subsubsection{AIML (Rule-Based)}
- Pattern matching with wildcards
- 150 hand-crafted patterns
- Instant inference (5ms)
- Memory: 10MB

\subsubsection{DialoGPT (Neural)}
- microsoft/DialoGPT-small
- 117M parameters
- Pre-trained on Reddit
- Inference: 250ms

\subsubsection{Hybrid (GPT-2 + Intent)}
- Intent: BART-large-MNLI (zero-shot)
- Generation: GPT-2 (124M parameters)
- Two-stage pipeline:
  1. Classify intent
  2. Generate conditioned on intent
- Inference: 280ms

\subsection{Evaluation Metrics}
- Classification: Accuracy, Precision, Recall, F1
- Generation: BLEU (1-4), ROUGE (1,2,L), METEOR
- Quality: Diversity, Length stats
- Error analysis: Categorization, confidence

\subsection{Experimental Setup}
- Hardware: CPU-based (no GPU)
- Software: Python 3.12, PyTorch 2.9.1, Transformers 4.57.3
- Cross-validation: 5-fold
- Random seed: 42 (reproducibility)
```

### Priority 2: Related Work (2 hours)

Review these 4 papers:

1. **Adamopoulou & Moussiades (2020)** - Overview of chatbot technology
2. **Laranjo et al. (2018)** - Healthcare conversational agents
3. **Ranoliya et al. (2017)** - University FAQ chatbots
4. **Medical ChatBot paper** - Domain-specific applications

Structure:
```latex
\section{Related Work}
\label{sec:related_work}

\subsection{Chatbot Technology Evolution}
Cite: Adamopoulou & Moussiades
- ELIZA to modern transformers
- Dialogue management strategies
- NLP techniques progression

\subsection{Healthcare Applications}
Cite: Laranjo et al.
- Clinical effectiveness
- User engagement
- Privacy concerns

\subsection{Domain-Specific Implementations}
Cite: Ranoliya, Medical ChatBot
- FAQ systems
- Pattern matching vs ML
- Hybrid approaches

\subsection{Gap in Literature}
- Limited architectural comparisons
- Insufficient error analysis
- Lack of hybrid approach studies
```

### Priority 3: Shorter Sections (2 hours)

**Conclusion** (400 words, 30 min)
```latex
- Summarize 3 architectures tested
- Key findings: hybrid superiority (F1: 0.83)
- Main insight: 60% errors from intent misclassification
- Practical implications
- Future directions
```

**Limitations** (300 words, 20 min)
```latex
- English-only
- Small models tested
- No human evaluation
- CPU-only experiments
```

**Ethical Statement** (400 words, 30 min)
```latex
- Mental health chatbots are NOT clinical tools
- Privacy and data protection
- Bias and fairness considerations
- Transparency requirements
```

**Acknowledgements** (100 words, 10 min)
```latex
- Course instructors
- Open-source projects used
- Dataset providers
```

### Priority 4: References (30 min)

```bibtex
@article{adamopoulou2020overview,
  title={An overview of chatbot technology},
  author={Adamopoulou, Eleni and Moussiades, Lefteris},
  ...
}

@article{laranjo2018conversational,
  title={Conversational agents in healthcare},
  author={Laranjo, Liliana and others},
  ...
}

% Add ~20-25 more references
```

---

## LaTeX Compilation

### Build Commands

```bash
cd NLP_Paper_Template

# First compilation
pdflatex main.tex

# Process references
bibtex main

# Resolve references (run twice)
pdflatex main.tex
pdflatex main.tex

# View PDF
start main.pdf  # Windows
```

### Common LaTeX Errors

**Error:** `Undefined control sequence`
- **Fix:** Check for typos in LaTeX commands

**Error:** `Citation undefined`
- **Fix:** Run `bibtex main` after first `pdflatex`

**Error:** `Missing $ inserted`
- **Fix:** Math mode needed: wrap with `$ ... $` or `\[ ... \]`

**Error:** `Overfull \hbox`
- **Fix:** LaTeX warning (not critical), adjust text or use `\sloppy`

---

## Writing Tips

### For Methodology

- Use past tense ("We implemented...", "The model was trained...")
- Be specific about versions (Python 3.12, PyTorch 2.9.1)
- Reference actual code structure (src/domain/services/...)
- Include implementation details others could replicate

### For Related Work

- Group papers by theme (chatbot tech, healthcare, domain-specific)
- Compare/contrast approaches
- Identify gaps that your work addresses
- Use present tense for established facts ("DialoGPT uses...")

### For Discussion

- Connect results to research questions (RQ1-RQ4)
- Explain WHY results occurred, not just WHAT
- Compare with related work quantitatively when possible
- Acknowledge limitations honestly

---

## Quality Checklist

### Before Submission

- [ ] All tables have captions and labels
- [ ] All figures referenced in text
- [ ] Consistent terminology throughout
- [ ] Citations formatted correctly
- [ ] Math notation consistent
- [ ] Acronyms defined at first use
- [ ] Section references work (\\ref{sec:...})
- [ ] PDF compiles without errors
- [ ] Proofread for typos/grammar
- [ ] Word count appropriate (~4000-5000 words)

---

## Time Estimate

| Section | Words | Time | Status |
|---------|-------|------|--------|
| Abstract | 200 | 30m | Done |
| Introduction | 1200 | 1.5h | Done |
| Related Work | 800 | 2h |  TODO |
| Methodology | 1500 | 2h |  TODO |
| Results | 1800 | 2h | Done |
| Discussion | 2400 | 2h | Done |
| Conclusion | 400 | 30m |  TODO |
| Limitations | 300 | 20m |  TODO |
| Ethics | 400 | 30m |  TODO |
| Acknowledgements | 100 | 10m |  TODO |
| References | - | 30m |  TODO |
| **Total** | **9100** | **12h** | **33% Done** |

**Completed:** 4 hours (33%)  
**Remaining:** 6-8 hours (67%)

---

## Resources

### LaTeX Reference

- [Overleaf Documentation](https://www.overleaf.com/learn)
- [LaTeX Tables Generator](https://www.tablesgenerator.com/)
- [ACL 2023 Style Guide](https://2023.aclweb.org/calls/style_and_formatting/)

### Citation Management

- [Google Scholar](https://scholar.google.com/) - Get BibTeX citations
- [ACL Anthology](https://aclanthology.org/) - NLP paper repository

### Writing Guides

- [How to Write a Paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/)
- [Academic Phrasebank](http://www.phrasebank.manchester.ac.uk/)

---

## Current Word Count

- Abstract: 200 words - Introduction: 1,200 words - Results: 1,800 words - Discussion: 2,400 words - **Total so far:** 5,600 words

**Target:** 8,000-10,000 words for complete paper

---

**Day 6 Status:** 33% Complete (4/11 sections)  
**Next:** Write Methodology (2h) → Related Work (2h) → Remaining sections (2h)
