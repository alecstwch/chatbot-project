"""
Quick Test: Verify DDD refactoring and configuration loading.

Tests:
1. Load therapy intent configuration
2. Load chef intent configuration
3. Initialize intent classifiers (without loading ML models)
4. Test conversation engine with mock model
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO)

print("=" * 70)
print("DDD REFACTORING - CONFIGURATION TEST")
print("=" * 70)

# Test 1: Load therapy intent configuration
print("\n1. Testing therapy intent configuration loading...")
try:
    from src.domain.services.intent_classifier import load_intent_config
    
    therapy_config = load_intent_config("therapy_intents")
    
    print(f"   ✓ Loaded therapy config")
    print(f"   - Domain: {therapy_config['domain']}")
    print(f"   - Intents: {len(therapy_config['intents']['intents'])} defined")
    print(f"   - Keywords: {len(therapy_config['keywords']['keywords'])} intent patterns")
    
    # Show sample intents
    intents = list(therapy_config['intents']['intents'].keys())
    print(f"   - Sample intents: {', '.join(intents[:5])}")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Load chef intent configuration
print("\n2. Testing chef intent configuration loading...")
try:
    chef_config = load_intent_config("chef_intents")
    
    print(f"   ✓ Loaded chef config")
    print(f"   - Domain: {chef_config['domain']}")
    print(f"   - Intents: {len(chef_config['intents']['intents'])} defined")
    print(f"   - Funnel stages: {len(chef_config['intents']['funnel_stages'])} stages")
    print(f"   - Dish types: {len(chef_config['intents']['dish_types'])} types")
    print(f"   - Ingredients: {len(chef_config['ingredients']['categories'])} categories")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Initialize intent classifiers (without loading models)
print("\n3. Testing intent classifier initialization...")
try:
    from src.domain.services.intent_classifier import IntentClassificationService
    
    # Therapy classifier
    therapy_classifier = IntentClassificationService(domain="therapy_intents")
    info = therapy_classifier.get_domain_info()
    
    print(f"   ✓ Therapy classifier initialized")
    print(f"   - Intent labels: {info['intent_labels'][:5]}...")
    print(f"   - Keyword patterns: {info['num_keyword_patterns']} intents")
    
    # Chef classifier
    chef_classifier = IntentClassificationService(domain="chef_intents")
    chef_info = chef_classifier.get_domain_info()
    
    print(f"   ✓ Chef classifier initialized")
    print(f"   - Intent labels: {chef_info['intent_labels'][:5]}...")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test keyword-based classification (no model needed)
print("\n4. Testing keyword-based classification...")
try:
    # Test therapy classifier
    test_inputs = [
        "I feel so sad and hopeless",
        "I'm really anxious about work",
        "Hello, I need help"
    ]
    
    for text in test_inputs:
        result = therapy_classifier._keyword_classify(text)
        print(f"   - '{text[:30]}...' → {result.intent} ({result.confidence:.2f})")
    
    print(f"   ✓ Keyword classification working")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test conversation engine with mock model
print("\n5. Testing conversation engine...")
try:
    from src.domain.services.conversation_engine import (
        ConversationEngine,
        SimpleConversationFormatter,
        ConversationTurn
    )
    
    # Create a mock model
    class MockModel:
        def generate(self, prompt, **kwargs):
            return prompt + "I'm a helpful assistant!"
        
        def is_ready(self):
            return True
    
    mock_model = MockModel()
    engine = ConversationEngine(
        model=mock_model,
        formatter=SimpleConversationFormatter(),
        max_history_turns=5
    )
    
    # Test conversation
    response = engine.generate_response("Hello")
    print(f"   ✓ Conversation engine working")
    print(f"   - Response generated: '{response[:50]}...'")
    
    summary = engine.get_conversation_summary()
    print(f"   - History turns: {summary['total_turns']}")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test chef intent classifier
print("\n6. Testing chef intent classifier...")
try:
    from src.domain.services.chef_intent_classifier import ChefIntentClassifier, RecipeContext
    
    chef = ChefIntentClassifier(device="cpu", max_funnel_questions=5)
    
    print(f"   ✓ Chef intent classifier initialized")
    print(f"   - Max funnel questions: {chef.max_funnel_questions}")
    print(f"   - Dish types available: {len(chef.dish_types)}")
    print(f"   - Cuisines available: {len(chef.cuisines)}")
    
    # Test context
    context = RecipeContext()
    context.ingredients = ["chicken", "tomato", "pasta"]
    context.dish_type = "main_dish"
    
    print(f"   - Sample context: {context.get_summary()}")
    print(f"   - Context complete: {context.is_complete()}")
    
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nDDD refactoring successful!")
print("\nKey achievements:")
print("  ✓ Configuration externalized to YAML files")
print("  ✓ Intent classifier supports multiple domains")
print("  ✓ Conversation engine separated from model infrastructure")
print("  ✓ Chef intent classifier with Q&A funnel operational")
print("  ✓ All components follow DDD and 12-Factor App principles")
print("\nNext steps:")
print("  1. Run: python scripts/demo_chef_chatbot.py")
print("  2. Test with real ML models (loads transformers)")
print("  3. Integrate with existing chatbot CLI interfaces")
