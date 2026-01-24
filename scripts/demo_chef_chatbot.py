"""
Demonstration: Master Chef Recipe Recommendation System.

Shows how the chef intent classifier works with a Q&A funnel to recommend recipes.
"""

import sys
from pathlib import Path

# Fix Windows console encoding for UTF-8 characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.domain.services.chef_intent_classifier import ChefIntentClassifier

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run chef chatbot demo."""
    print("MASTER CHEF RECIPE RECOMMENDATION DEMO")
    print("\nThis demo shows the conversational Q&A funnel for recipe recommendation.")
    print("Maximum 5 questions to collect your preferences.\n")
    
    # Initialize chef classifier
    print("Initializing Master Chef assistant...")
    chef = ChefIntentClassifier(
        model_name="facebook/bart-large-mnli",
        device="cpu",  # Use CPU for demo
        max_funnel_questions=5
    )
    
    print("Loading AI model (this may take a moment)...\n")
    chef.load_model()
    
    print("Chef assistant ready!\n")
    
    # Demo conversation with all 5 questions
    demo_responses = [
        "eggs, flour, milk, butter, chicken, tomatoes, onions, garlic",  # Ingredients
        "none",  # Dietary constraints/restrictions
        "main dish",  # Dish type
        "Italian",  # Cuisine preference
        "moderate"  # Complexity
    ]
    
    # Start conversation
    funnel_complete = False
    question_count = 0
    response_index = 0
    
    # Get initial greeting
    next_question, _, _ = chef.process_user_input("Hello")
    print(f"\nChef: {next_question}\n")
    
    # Automated funnel demo
    while not funnel_complete and question_count < chef.max_funnel_questions:
        if response_index < len(demo_responses):
            user_input = demo_responses[response_index]
            response_index += 1
        else:
            break
        
        print(f"You: {user_input}")
        
        # Process input through funnel
        response, intent, funnel_complete = chef.process_user_input(user_input)
        
        print(f"\n[Detected intent: {intent.intent} (confidence: {intent.confidence:.2f})]")
        print(f"Chef: {response}\n")
        
        question_count += 1
        
        if funnel_complete:
            print("RECIPE RECOMMENDATIONS COMPLETE")
            
            # Show collected context
            context = chef.get_context()
            print(f"\nCollected all preferences through the funnel!\n")
            print(f"Summary: {context.get_summary()}\n")
            
            # Show detailed breakdown
            print("Detailed Preferences:")
            print(f"   Ingredients: {', '.join(context.ingredients) if context.ingredients else 'Not specified'}")
            print(f"   Dish Type: {context.dish_type or 'Not specified'}")
            print(f"   Dietary Constraints: {', '.join(context.constraints) if context.constraints else 'None'}")
            print(f"   Cuisine: {context.cuisine or 'Not specified'}")
            print(f"   Complexity: {context.complexity or 'Not specified'}")
            print()
            
            # In a real app, this would query a recipe database
            print("In a production system, this would:")
            print("1. Query a recipe database with collected preferences")
            print("2. Match recipes based on ingredient availability")
            print("3. Filter by dietary constraints and dish type")
            print("4. Rank by cuisine preference and complexity")
            print("5. Provide step-by-step cooking instructions")
            print("\nExample Recommendations:")
            print("   Gluten-Free Italian Chicken Primavera (Medium)")
            print("   Italian Chicken & Tomato Risotto (Medium)")
            print("   Garlic Butter Chicken with Roasted Vegetables (Medium)\n")
            
            break
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
