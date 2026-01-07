# 7-Day Chatbot Sprint - Quick Start Guide

## Two Chatbots in 7 Days Using Open Source NLP

---

## Project At a Glance

| Aspect | Details |
|--------|---------|
| **Duration** | 7 days (intensive) |
| **Chatbots** | 2 (Psychotherapy + General Conversational) |
| **Cost** | $0 (100% open source) |
| **Model Types** | Traditional (AIML), Neural (DialoGPT), Transformer (DistilBERT/GPT-2) |
| **Primary Focus** | NLP techniques + speed |

---

## Day-by-Day Breakdown

###  Day 1: Environment & Data (Monday)
**Goal:** Working development environment + clean datasets

```bash
# 1. Create environment (Python 3.11.9 for compatibility)
py -3.11 -m venv chatbot-env
chatbot-env\Scripts\activate  # Windows
# source chatbot-env/bin/activate  # Linux/Mac

# 2. Install everything (ChatterBot excluded - incompatible with Python 3.11+)
pip install transformers datasets torch nltk spacy scikit-learn
pip install python-aiml  # Rule-based chatbot framework
pip install pandas matplotlib seaborn jupyter plotly

# 3. Download NLTK data (specific packages for NLTK 3.9.2)
python -c "import nltk; nltk.download(['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger'])"

# 4. Download spaCy model
python -m spacy download en_core_web_sm
```

**Load Datasets:**
```python
from datasets import load_dataset

# Psychotherapy dataset
therapy_data = load_dataset("Amod/mental_health_counseling_conversations")

# General conversation dataset  
dialog_data = load_dataset("daily_dialog")

# Save locally for faster access
therapy_data.save_to_disk("./data/therapy")
dialog_data.save_to_disk("./data/dialogs")
```

**Quick EDA:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Convert to DataFrame
df_therapy = pd.DataFrame(therapy_data['train'])
df_dialog = pd.DataFrame(dialog_data['train'])

# Basic stats
print(f"Therapy conversations: {len(df_therapy)}")
print(f"General dialogs: {len(df_dialog)}")

# Length distribution
df_therapy['length'] = df_therapy['text'].str.len()
df_therapy['length'].hist(bins=50)
plt.title("Therapy Message Length Distribution")
plt.savefig("therapy_length_dist.png")
```

[DONE] **End of Day 1:** You have clean data and working environment

**[DONE] COMPLETED:** Project restructured with DDD architecture, configuration system with .env, and 27 passing unit tests with 100% coverage. See [RESTRUCTURING_COMPLETE.md](RESTRUCTURING_COMPLETE.md) for details.

---

###  Day 2: Psychotherapy Bot (Traditional/AIML) (Tuesday)
**Goal:** Working rule-based therapy chatbot

**Create AIML Knowledge Base:**
```xml
<!-- therapy.aiml -->
<aiml version="1.0.1" encoding="UTF-8"?>
    <category>
        <pattern>I FEEL *</pattern>
        <template>
            Can you tell me more about why you feel <star/>?
        </template>
    </category>
    
    <category>
        <pattern>I AM DEPRESSED</pattern>
        <template>
            I'm sorry to hear that. Depression is challenging. 
            How long have you been feeling this way?
        </template>
    </category>
    
    <category>
        <pattern>* ANXIOUS *</pattern>
        <template>
            Anxiety can be overwhelming. What helps you feel calmer?
        </template>
    </category>
</aiml>
```

**Python Implementation:**
```python
import aiml

# Create kernel
kernel = aiml.Kernel()

# Load AIML files
kernel.learn("therapy.aiml")

# Test chatbot
def chat():
    print("Therapy Bot: Hello, how are you feeling today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = kernel.respond(user_input)
        print(f"Therapy Bot: {response}")

chat()
```

**NLP Processing (Using DDD Architecture):**
```python
# Import from domain services
from src.domain.services.text_preprocessor import TextPreprocessingService

# Initialize service
preprocessor = TextPreprocessingService(
    remove_stopwords=True,
    apply_stemming=False,
    apply_lemmatization=True
)

# Preprocess text
text = "I am feeling very anxious today"
processed = preprocessor.preprocess(text)
print(processed)  # Output: "feeling anxious today"

# Batch preprocessing
texts = ["I feel sad", "I am worried", "Help me please"]
processed_batch = preprocessor.batch_preprocess(texts)
print(processed_batch)
```

[DONE] **End of Day 2:** Working AIML therapy chatbot with NLP preprocessing

---

###  Day 3: General Chatbot (Neural/DialoGPT) (Wednesday)
**Goal:** Working neural conversational chatbot

**Load Pre-trained DialoGPT:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

def chat_with_bot(user_input, chat_history_ids=None):
    # Encode input
    new_input_ids = tokenizer.encode(
        user_input + tokenizer.eos_token, 
        return_tensors='pt'
    )
    
    # Append to chat history
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids
    
    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9
    )
    
    # Decode response
    response = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
        skip_special_tokens=True
    )
    
    return response, chat_history_ids

# Test conversation
chat_history = None
print("General Bot: Hi! Let's chat!")
while True:
    user_msg = input("You: ")
    if user_msg.lower() == 'quit':
        break
    response, chat_history = chat_with_bot(user_msg, chat_history)
    print(f"Bot: {response}")
```

**Optional: Fine-tune on Daily Dialog (if time permits)**
```python
from transformers import Trainer, TrainingArguments

# Prepare dataset
def tokenize_function(examples):
    return tokenizer(examples['dialog'], truncation=True, padding='max_length')

tokenized_datasets = dialog_data.map(tokenize_function, batched=True)

# Training arguments (minimal training)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,  # Just 1 epoch for speed
    per_device_train_batch_size=4,
    save_steps=1000,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)

trainer.train()
```

[DONE] **End of Day 3:** Working DialoGPT chatbot (pre-trained or fine-tuned)

---

###  Day 4: Transformer Integration (Thursday)
**Goal:** Add transformer-based intent classification and generation

**Intent Classification with DistilBERT:**
```python
from transformers import pipeline

# Load pre-trained classifier
classifier = pipeline("text-classification", model="distilbert-base-uncased")

# For therapy bot: classify intent
def classify_intent(text):
    intents = {
        'depression': ['depressed', 'sad', 'hopeless'],
        'anxiety': ['anxious', 'worried', 'nervous'],
        'general': ['hello', 'hi', 'help']
    }
    
    text_lower = text.lower()
    for intent, keywords in intents.items():
        if any(kw in text_lower for kw in keywords):
            return intent
    return 'general'

# Use in chatbot
user_input = "I'm feeling very anxious"
intent = classify_intent(user_input)
print(f"Detected intent: {intent}")
```

**Response Generation with GPT-2:**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 small
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_response(prompt):
    inputs = gpt2_tokenizer.encode(prompt, return_tensors='pt')
    outputs = gpt2_model.generate(
        inputs, 
        max_length=100, 
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
response = generate_response("How can I manage anxiety?")
print(response)
```

**Hybrid Chatbot (AIML + Transformer fallback):**
```python
class HybridTherapyBot:
    def __init__(self):
        self.aiml_kernel = aiml.Kernel()
        self.aiml_kernel.learn("therapy.aiml")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    def respond(self, user_input):
        # Try AIML first
        aiml_response = self.aiml_kernel.respond(user_input)
        
        # If no good AIML match, use GPT-2
        if not aiml_response or len(aiml_response) < 10:
            prompt = f"Therapist response to: {user_input}\nTherapist:"
            return self.generate_gpt2_response(prompt)
        
        return aiml_response
    
    def generate_gpt2_response(self, prompt):
        inputs = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.gpt2_model.generate(inputs, max_length=100)
        return self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Use hybrid bot
bot = HybridTherapyBot()
print(bot.respond("I feel anxious"))  # AIML response
print(bot.respond("What's the meaning of life?"))  # GPT-2 fallback
```

[DONE] **End of Day 4:** Both chatbots using all 3 model types

**[DONE] COMPLETED:** All 3 model types integrated with intent classification and GPT-2 generation. See [DAY4.md](docs/DAY4.md) for complete implementation details, including:
- Intent Classification Service (zero-shot with BART)
- Response Generation Service (GPT-2)
- Hybrid Chatbot (AIML + GPT-2 + Intent)
- Transformer-Enhanced DialoGPT (with intent awareness)
- Comprehensive unit tests and CLI interfaces
- Demo script showcasing all approaches

**Evaluation Metrics:**
```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

# 1. BLEU Score for generation quality
def calculate_bleu(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens)

# Test
ref = "I understand you're feeling anxious"
gen = "I can see you feel worried"
print(f"BLEU: {calculate_bleu(ref, gen)}")

# 2. Intent Classification Accuracy
def evaluate_intent_classification(test_data):
    predictions = []
    labels = []
    
    for item in test_data:
        pred = classify_intent(item['text'])
        predictions.append(pred)
        labels.append(item['label'])
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return acc, f1

# 3. Response Appropriateness (manual evaluation)
test_conversations = [
    ("I'm feeling depressed", "expected: empathetic response"),
    ("Hello", "expected: greeting"),
]

for user_msg, expectation in test_conversations:
    response = bot.respond(user_msg)
    print(f"Input: {user_msg}")
    print(f"Response: {response}")
    print(f"Expectation: {expectation}")
    print("---")
```

**Error Analysis:**
```python
# Collect failure cases
errors = []

for test_input in test_dataset:
    response = bot.respond(test_input)
    if not is_appropriate_response(response, test_input):
        errors.append({
            'input': test_input,
            'response': response,
            'issue': classify_error_type(response)
        })

# Categorize errors
error_types = {}
for error in errors:
    error_type = error['issue']
    error_types[error_type] = error_types.get(error_type, 0) + 1

print("Error Distribution:")
for error_type, count in error_types.items():
    print(f"{error_type}: {count}")
```

**Explainability with LIME:**
```python
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['depression', 'anxiety', 'general'])

def predict_proba(texts):
    # Your classifier's prediction function
    # Return probabilities for each class
    pass

# Explain a prediction
exp = explainer.explain_instance(
    "I feel very anxious today", 
    predict_proba,
    num_features=10
)

exp.show_in_notebook()
```

**Attention Visualization (for transformers):**
```python
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

def visualize_attention(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
    
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    
    # Get attention weights
    attention = outputs.attentions[-1][0].detach().numpy()
    
    # Plot heatmap
    plt.imshow(attention[0], cmap='viridis')
    plt.colorbar()
    plt.title("Attention Weights")
    plt.savefig("attention_viz.png")

visualize_attention("I am feeling anxious")
```

[DONE] **End of Day 5:** Complete evaluation with metrics, errors, and explainability

---

###  Day 6: Paper Writing (Saturday)
**Goal:** Complete LaTeX paper

**Suggested Writing Order:**
1. **Methodology (2h)** - Easiest, you know what you did
2. **Results (2h)** - Tables and figures ready from Day 5
3. **Introduction (1h)** - Set context and motivation
4. **Related Work (2h)** - Review the 4 provided papers
5. **Discussion (1h)** - Interpret results
6. **Abstract (30min)** - Summarize everything
7. **Conclusion, Limitations, Ethics (1h)** - Final sections

**Quick LaTeX Setup:**
```latex
% Key results table
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
Model & BLEU & F1-Score \\
\hline
AIML (Rule-based) & 0.42 & 0.68 \\
DialoGPT (Neural) & 0.71 & 0.84 \\
Hybrid (AIML+GPT-2) & 0.79 & 0.89 \\
\hline
\end{tabular}
\caption{Performance comparison across model types}
\end{table}
```

**Figures to Include:**
- Dataset statistics (from Day 1)
- Preprocessing comparison (Day 2)
- Model architecture diagram
- Performance comparison charts (Day 5)
- Error analysis distribution (Day 5)
- Attention visualization (Day 5)

[DONE] **End of Day 6:** Complete paper draft in LaTeX

---

###  Day 7: Finalization (Sunday)
**Goal:** Polish everything and submit

**Morning (4h): Paper Polish**
- [ ] Proofread entire paper
- [ ] Check all references
- [ ] Verify figures are clear
- [ ] Format tables properly
- [ ] Run spell check
- [ ] Compile PDF without errors

**Afternoon (3h): Presentation**
```
Slide 1: Title
Slide 2: Motivation (why therapy + general chatbots?)
Slide 3: Research Questions
Slide 4: Dataset Overview
Slide 5: Preprocessing Pipeline
Slide 6: Model 1 - AIML (Rule-based)
Slide 7: Model 2 - DialoGPT (Neural)
Slide 8: Model 3 - Hybrid (Transformer)
Slide 9: Results Comparison
Slide 10: Error Analysis
Slide 11: Demo (live or screenshots)
Slide 12: Conclusions & Future Work
```

**Evening (1h): Final Checks**
- [ ] Test both chatbots one more time
- [ ] Clean up code repository
- [ ] Write README.md
- [ ] Create requirements.txt
- [ ] Zip everything for submission

[DONE] **End of Day 7:** Ready to submit!

---

## Essential Code Snippets to Reuse

### 1. Quick Data Loading
```python
from datasets import load_dataset

therapy = load_dataset("Amod/mental_health_counseling_conversations")
dialogs = load_dataset("daily_dialog")
```

### 2. Text Preprocessing
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return ' '.join(tokens)
```

### 3. Load Any HuggingFace Model
```python
from transformers import pipeline

# For classification
classifier = pipeline("text-classification", model="distilbert-base-uncased")

# For generation
generator = pipeline("text-generation", model="gpt2")

# For conversation
chatbot = pipeline("conversational", model="microsoft/DialoGPT-small")
```

### 4. Quick Evaluation
```python
from nltk.translate.bleu_score import sentence_bleu

def evaluate(reference, generated):
    return sentence_bleu([reference.split()], generated.split())
```

---

## Resources for 7 Days

### Must-Read Papers (4 total, ~2h each = 8h total)
1. Adamopoulou & Moussiades - "An Overview of Chatbot Technology"
2. Laranjo et al. - "Conversational agents in healthcare"
3. Ranoliya et al. - "Chatbot for university related FAQs"
4. Medical ChatBot paper

### Essential Documentation
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [NLTK Documentation](https://www.nltk.org/)
- [PyTorch Tutorial](https://pytorch.org/tutorials/)
- [Python AIML](http://www.aiml.foundation/)

### Datasets (Pre-downloaded)
- `Amod/mental_health_counseling_conversations`
- `daily_dialog`

### Tools
- **Google Colab** (for GPU training)
- **VS Code** (for coding)
- **Overleaf** (for LaTeX)
- **GitHub** (for version control)

---

## Emergency Shortcuts (If Running Behind)

### Day 3 Shortcut
Skip fine-tuning DialoGPT. Use pre-trained directly.

### Day 4 Shortcut  
Use zero-shot classification instead of training:
```python
classifier = pipeline("zero-shot-classification")
classifier("I feel anxious", candidate_labels=["depression", "anxiety", "general"])
```

### Day 5 Shortcut
Manual evaluation on 20 test cases instead of full metrics.

### Day 6 Shortcut
Focus on: Abstract, Intro, Methodology, Results, Conclusion.
Skip: Extensive Related Work, Appendices.

---

## Success Checklist

By end of 7 days, you should have:
- [DONE] Two working chatbots (therapy + general)
- [DONE] All 3 model types demonstrated (AIML, Neural, Transformer)
- [DONE] Complete LaTeX paper (all sections)
- [DONE] Presentation slides
- [DONE] Code repository with README
- [DONE] Evaluation results with metrics
- [DONE] Demo-ready chatbots

---

## Team Coordination (if working in pairs)

### Parallel Track A (Member 1)
- Day 1: Data prep + Therapy dataset
- Day 2: AIML therapy bot
- Day 4: DistilBERT integration
- Day 6: Write Intro, Related Work, Discussion

### Parallel Track B (Member 2)  
- Day 1: Data prep + Dialog dataset
- Day 3: DialoGPT chatbot
- Day 4: GPT-2 integration
- Day 6: Write Methodology, Results, Conclusion

### Together
- Day 5: Evaluation & Analysis
- Day 7: Finalization

---

**You got this! 7 days to two amazing chatbots! **
