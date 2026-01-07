"""
Quick test script to verify chatbot-env setup
Run this to ensure all packages are working correctly
"""

import sys

def test_imports():
    """Test all critical imports"""
    print("Testing package imports...\n")
    
    packages = {
        "NLTK": "nltk",
        "spaCy": "spacy",
        "Transformers": "transformers",
        "PyTorch": "torch",
        "Datasets": "datasets",
        "scikit-learn": "sklearn",
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "AIML": "aiml",
        "LIME": "lime",
        "SHAP": "shap",
        "Gensim": "gensim",
        "Sentence Transformers": "sentence_transformers"
    }
    
    failed = []
    
    for name, module in packages.items():
        try:
            __import__(module)
            print(f"[OK] {name}")
        except ImportError as e:
            print(f"[FAIL] {name} - {e}")
            failed.append(name)
    
    print(f"\n{'='*50}")
    if not failed:
        print("[OK] ALL PACKAGES IMPORTED SUCCESSFULLY!")
    else:
        print(f"[FAIL] Failed to import: {', '.join(failed)}")
    print(f"{'='*50}\n")
    
    return len(failed) == 0


def test_nltk_data():
    """Test NLTK data availability"""
    print("Testing NLTK data...\n")
    
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    try:
        # Test tokenization
        text = "Hello, how are you doing today?"
        tokens = word_tokenize(text)
        print(f"[OK] Tokenization works: {tokens[:3]}...")
        
        # Test stopwords
        stop_words = stopwords.words('english')
        print(f"[OK] Stopwords loaded: {len(stop_words)} words")
        
        # Test lemmatization
        lemmatizer = WordNetLemmatizer()
        lemma = lemmatizer.lemmatize("running", pos='v')
        print(f"[OK] Lemmatization works: 'running' -> '{lemma}'")
        
        print("\n[OK] All NLTK data tests passed!\n")
        return True
    except Exception as e:
        print(f"\n[FAIL] NLTK data test failed: {e}\n")
        return False


def test_spacy():
    """Test spaCy model"""
    print("Testing spaCy model...\n")
    
    import spacy
    
    try:
        nlp = spacy.load('en_core_web_sm')
        doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
        
        print(f"[OK] spaCy model loaded: en_core_web_sm")
        print(f"[OK] Entities found: {[(ent.text, ent.label_) for ent in doc.ents]}")
        
        print("\n[OK] spaCy test passed!\n")
        return True
    except Exception as e:
        print(f"\n[FAIL] spaCy test failed: {e}\n")
        return False


def test_transformers():
    """Test Transformers library"""
    print("Testing Transformers...\n")
    
    from transformers import pipeline
    
    try:
        # Test sentiment analysis (downloads small model)
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = classifier("I love this chatbot project!")
        
        print(f"[OK] Sentiment analysis works: {result[0]}")
        
        print("\n[OK] Transformers test passed!\n")
        return True
    except Exception as e:
        print(f"\n[FAIL] Transformers test failed: {e}\n")
        print("Note: First run downloads models, this is normal\n")
        return False


def test_pytorch():
    """Test PyTorch"""
    print("Testing PyTorch...\n")
    
    import torch
    
    try:
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        
        # Create a simple tensor
        x = torch.tensor([1, 2, 3])
        print(f"[OK] Tensor creation works: {x}")
        
        print("\n[OK] PyTorch test passed!\n")
        return True
    except Exception as e:
        print(f"\n[FAIL] PyTorch test failed: {e}\n")
        return False


def test_aiml():
    """Test python-aiml"""
    print("Testing python-aiml...\n")
    
    import aiml
    
    try:
        kernel = aiml.Kernel()
        print(f"[OK] AIML kernel created")
        
        # Simple test pattern
        kernel.learn("""
        <aiml version="1.0.1" encoding="UTF-8">
            <category>
                <pattern>HELLO</pattern>
                <template>Hi there!</template>
            </category>
        </aiml>
        """)
        
        response = kernel.respond("hello")
        print(f"[OK] AIML pattern matching works: 'hello' -> '{response}'")
        
        print("\n[OK] python-aiml test passed!\n")
        return True
    except Exception as e:
        print(f"\n[FAIL] python-aiml test failed: {e}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("CHATBOT-ENV SETUP VERIFICATION")
    print("="*50 + "\n")
    
    print(f"Python version: {sys.version}\n")
    print(f"Python executable: {sys.executable}\n")
    
    results = []
    
    results.append(("Package Imports", test_imports()))
    results.append(("NLTK Data", test_nltk_data()))
    results.append(("spaCy", test_spacy()))
    results.append(("PyTorch", test_pytorch()))
    results.append(("python-aiml", test_aiml()))
    
    # Transformers test is optional (downloads models)
    print("\nNote: Transformers test downloads models on first run...")
    try:
        results.append(("Transformers", test_transformers()))
    except KeyboardInterrupt:
        print("\nTransformers test skipped by user\n")
        results.append(("Transformers", None))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50 + "\n")
    
    for test_name, result in results:
        if result is True:
            status = "[OK] PASS"
        elif result is False:
            status = "[FAIL] FAIL"
        else:
            status = "⊘ SKIP"
        print(f"{status:8} - {test_name}")
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    print(f"\n{'='*50}")
    print(f"Total: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n ENVIRONMENT IS READY TO USE! ")
        print("\nYou can start working on your chatbot project!")
    else:
        print("\n[WARNING] SOME TESTS FAILED")
        print("Please check the error messages above")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
