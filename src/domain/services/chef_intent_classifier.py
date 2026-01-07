"""
Master Chef Intent Classifier and Recipe Recommendation System.

Implements a conversational Q&A funnel (max 5 questions) to collect:
- Available ingredients (5-10)
- Dietary constraints (allergens, restrictions)
- Dish type (snack, soup, main, dessert, etc.)
- Cuisine preference
- Complexity/time constraints

Then provides targeted recipe recommendations.
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from src.domain.services.intent_classifier import IntentClassificationService, IntentPrediction

logger = logging.getLogger(__name__)


class FunnelStage(Enum):
    """Recipe recommendation funnel stages."""
    GREETING = 0
    INGREDIENTS = 1
    CONSTRAINTS = 2
    DISH_TYPE = 3
    CUISINE = 4
    COMPLEXITY = 5
    RECIPE_GENERATION = 6


@dataclass
class RecipeContext:
    """
    Context collected through the Q&A funnel.
    
    Attributes:
        ingredients: List of available ingredients (5-10)
        constraints: Dietary constraints, allergies, health goals
        dish_type: Type of dish (snack, soup, main_dish, dessert, etc.)
        cuisine: Preferred cuisine style
        complexity: Time/skill level (quick, moderate, complex)
        additional_preferences: Any other user preferences
    """
    ingredients: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    dish_type: Optional[str] = None
    cuisine: Optional[str] = None
    complexity: Optional[str] = None
    additional_preferences: Dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Check if all required information is collected through the funnel."""
        return (
            len(self.ingredients) >= 3 and
            self.dish_type is not None and
            self.cuisine is not None and
            self.complexity is not None
        )
    
    def get_summary(self) -> str:
        """Get human-readable summary of context."""
        parts = []
        
        if self.ingredients:
            parts.append(f"Ingredients: {', '.join(self.ingredients[:5])}")
            if len(self.ingredients) > 5:
                parts[-1] += f" (+{len(self.ingredients) - 5} more)"
        
        if self.constraints:
            parts.append(f"Constraints: {', '.join(self.constraints)}")
        
        if self.dish_type:
            parts.append(f"Dish type: {self.dish_type}")
        
        if self.cuisine:
            parts.append(f"Cuisine: {self.cuisine}")
        
        if self.complexity:
            parts.append(f"Complexity: {self.complexity}")
        
        return " | ".join(parts) if parts else "No context yet"


@dataclass
class RecipeRecommendation:
    """
    Recipe recommendation result.
    
    Attributes:
        name: Recipe name
        ingredients_needed: List of required ingredients
        missing_ingredients: Ingredients user doesn't have
        steps: Cooking steps
        prep_time: Preparation time in minutes
        difficulty: Difficulty level
        cuisine: Cuisine type
        dish_type: Type of dish
    """
    name: str
    ingredients_needed: List[str]
    missing_ingredients: List[str]
    steps: List[str]
    prep_time: int
    difficulty: str
    cuisine: str
    dish_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'ingredients_needed': self.ingredients_needed,
            'missing_ingredients': self.missing_ingredients,
            'num_steps': len(self.steps),
            'prep_time': f"{self.prep_time} minutes",
            'difficulty': self.difficulty,
            'cuisine': self.cuisine,
            'dish_type': self.dish_type
        }


class ChefIntentClassifier:
    """
    Chef-specific intent classifier with Q&A funnel for recipe recommendation.
    
    Implements a max 5-question funnel to collect recipe requirements,
    then generates targeted recommendations based on available ingredients
    and user constraints.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
        max_funnel_questions: int = 5
    ):
        """
        Initialize chef intent classifier.
        
        Args:
            model_name: HuggingFace model for zero-shot classification
            device: Device to run on
            max_funnel_questions: Maximum questions in funnel (default 5)
        """
        # Base intent classifier for chef domain
        self.intent_classifier = IntentClassificationService(
            domain="chef_intents",
            model_name=model_name,
            device=device
        )
        
        self.max_funnel_questions = max_funnel_questions
        self.current_stage = FunnelStage.GREETING
        self.questions_asked = 0
        self.recipe_context = RecipeContext()
        
        # Load chef-specific config
        self.config = self.intent_classifier.config
        self.funnel_config = self.config['intents'].get('funnel_stages', {})
        self.dish_types = self.config['intents'].get('dish_types', {})
        self.cuisines = self.config['intents'].get('cuisines', [])
        
        logger.info("Chef intent classifier initialized")
        logger.info(f"Max funnel questions: {max_funnel_questions}")
    
    def load_model(self) -> None:
        """Load the underlying intent classification model."""
        self.intent_classifier.load_model()
    
    def process_user_input(self, user_input: str) -> Tuple[str, IntentPrediction, bool]:
        """
        Process user input through the funnel.
        
        Args:
            user_input: User's message
            
        Returns:
            Tuple of (next_question, intent_prediction, funnel_complete)
        """
        # Classify intent
        intent_pred = self.intent_classifier.classify(user_input)
        
        # Update context based on intent and current stage
        self._update_context(user_input, intent_pred)
        
        # Determine next action - check completion AFTER updating context
        if self.recipe_context.is_complete():
            return self._generate_recipe_response(), intent_pred, True
        elif self.questions_asked >= self.max_funnel_questions:
            # Hit question limit but haven't completed - generate response anyway
            return self._generate_recipe_response(), intent_pred, True
        else:
            next_question = self._get_next_question()
            self.questions_asked += 1
            return next_question, intent_pred, False
            return next_question, intent_pred, False
    
    def _update_context(self, user_input: str, intent: IntentPrediction) -> None:
        """
        Update recipe context based on user input and detected intent.
        
        Args:
            user_input: User's message
            intent: Detected intent
        """
        intent_name = intent.intent
        text_lower = user_input.lower()
        
        logger.info(f"_update_context: input='{user_input}', intent={intent_name}, current_stage={self.current_stage}")
        
        # IMPORTANT: In funnel mode, ONLY use current_stage to determine which extraction to perform
        # This ensures we process answers in the correct order regardless of intent detection accuracy
        
        # Extract ingredients
        if self.current_stage == FunnelStage.INGREDIENTS:
            ingredients = self._extract_ingredients(user_input)
            self.recipe_context.ingredients.extend(ingredients)
        
        # Extract constraints
        elif self.current_stage == FunnelStage.CONSTRAINTS:
            constraints = self._extract_constraints(user_input)
            self.recipe_context.constraints.extend(constraints)
        
        # Identify dish type
        elif self.current_stage == FunnelStage.DISH_TYPE:
            dish_type = self._extract_dish_type(user_input)
            if dish_type:
                self.recipe_context.dish_type = dish_type
        
        # Identify cuisine
        elif self.current_stage == FunnelStage.CUISINE:
            cuisine = self._extract_cuisine(user_input)
            if cuisine:
                self.recipe_context.cuisine = cuisine
        
        # Identify complexity
        elif self.current_stage == FunnelStage.COMPLEXITY:
            complexity = self._extract_complexity(user_input)
            logger.info(f"Extracting complexity: '{user_input}' -> '{complexity}'")
            if complexity:
                self.recipe_context.complexity = complexity
                logger.info(f"Set complexity to: {self.recipe_context.complexity}")
    
    def _extract_ingredients(self, text: str) -> List[str]:
        """Extract ingredients from user text."""
        # Simple extraction - look for common ingredients
        ingredients = []
        text_lower = text.lower()
        
        # Get ingredient list from config
        if 'categories' in self.config.get('ingredients', {}):
            for category, items in self.config['ingredients']['categories'].items():
                if isinstance(items, dict):
                    for subcategory, ingredient_list in items.items():
                        if isinstance(ingredient_list, list):
                            for ing in ingredient_list:
                                if ing.lower() in text_lower:
                                    ingredients.append(ing)
                elif isinstance(items, list):
                    for ing in items:
                        if ing.lower() in text_lower:
                            ingredients.append(ing)
        
        # Deduplicate
        return list(set(ingredients))
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract dietary constraints from user text."""
        constraints = []
        text_lower = text.lower()
        
        constraint_config = self.config['intents'].get('dietary_constraints', {})
        
        # Check allergies
        for allergen in constraint_config.get('allergies', []):
            if allergen.lower() in text_lower:
                constraints.append(f"no_{allergen}")
        
        # Check dietary restrictions
        for restriction in constraint_config.get('dietary_restrictions', []):
            if restriction.lower() in text_lower:
                constraints.append(restriction)
        
        # Check health goals
        for goal in constraint_config.get('health_goals', []):
            if goal.lower() in text_lower:
                constraints.append(goal)
        
        return list(set(constraints))
    
    def _extract_dish_type(self, text: str) -> Optional[str]:
        """Extract dish type from user text."""
        text_lower = text.lower()
        
        for dish_type in self.dish_types.keys():
            if dish_type.replace('_', ' ') in text_lower or dish_type in text_lower:
                return dish_type
        
        return None
    
    def _extract_cuisine(self, text: str) -> Optional[str]:
        """Extract cuisine preference from user text."""
        text_lower = text.lower()
        
        for cuisine in self.cuisines:
            if cuisine.lower() in text_lower:
                return cuisine
        
        return None
    
    def _extract_complexity(self, text: str) -> Optional[str]:
        """Extract complexity level from user text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['quick', 'fast', 'easy', 'simple', '20', '15', 'minutes']):
            return 'quick'
        elif any(word in text_lower for word in ['moderate', 'medium', 'average', 'normal', '30', '40']):
            return 'moderate'
        elif any(word in text_lower for word in ['complex', 'gourmet', 'advanced', 'hour', 'difficult']):
            return 'complex'
        else:
            # Default to moderate if unclear
            return 'moderate'
    
    def _get_next_question(self) -> str:
        """Get the next question based on current funnel stage."""
        if self.current_stage == FunnelStage.GREETING:
            self.current_stage = FunnelStage.INGREDIENTS
            return "Hello! I'm your culinary assistant. What main ingredients do you have available? (Please list 5-10)"
        
        elif self.current_stage == FunnelStage.INGREDIENTS:
            self.current_stage = FunnelStage.CONSTRAINTS
            return "Great! Do you have any dietary restrictions, allergies, or health goals? (e.g., vegetarian, nut-free, low-calorie)"
        
        elif self.current_stage == FunnelStage.CONSTRAINTS:
            self.current_stage = FunnelStage.DISH_TYPE
            return "What type of dish are you looking for? (e.g., main dish, soup, salad, dessert, snack)"
        
        elif self.current_stage == FunnelStage.DISH_TYPE:
            self.current_stage = FunnelStage.CUISINE
            return "Any cuisine preference? (e.g., Italian, Mexican, Asian, Mediterranean)"
        
        elif self.current_stage == FunnelStage.CUISINE:
            self.current_stage = FunnelStage.COMPLEXITY
            return "How much time do you have for cooking? (quick, moderate, or complex)"
        
        else:
            return "Let me find some recipes for you!"
    
    def _generate_recipe_response(self) -> str:
        """
        Generate recipe recommendations based on collected context.
        
        Returns:
            Response with recipe recommendations
        """
        # In a real implementation, this would query a recipe database
        # For now, generate a mock response
        
        response = "Based on your preferences:\n"
        response += f"{self.recipe_context.get_summary()}\n\n"
        response += "Here are some recipe recommendations:\n\n"
        
        # Mock recommendations
        if self.recipe_context.dish_type == 'main_dish':
            if 'chicken' in self.recipe_context.ingredients:
                response += "1. **Chicken Stir-Fry** - Quick and healthy\n"
                response += "   Uses: chicken, vegetables, soy sauce\n"
                response += "   Time: 20 minutes\n\n"
            
            if 'pasta' in self.recipe_context.ingredients:
                response += "2. **Pasta Primavera** - Vegetarian option\n"
                response += "   Uses: pasta, vegetables, olive oil\n"
                response += "   Time: 25 minutes\n\n"
        
        elif self.recipe_context.dish_type == 'dessert':
            response += "1. **Simple Chocolate Mousse**\n"
            response += "   Time: 15 minutes (+ 2 hours chilling)\n\n"
        
        else:
            response += "1. **Custom Recipe** based on your ingredients\n"
            response += f"   Using: {', '.join(self.recipe_context.ingredients[:5])}\n\n"
        
        response += "Would you like detailed instructions for any of these?"
        
        return response
    
    def reset(self) -> None:
        """Reset the funnel state."""
        self.current_stage = FunnelStage.GREETING
        self.questions_asked = 0
        self.recipe_context = RecipeContext()
        logger.info("Chef funnel reset")
    
    def get_context(self) -> RecipeContext:
        """Get current recipe context."""
        return self.recipe_context
    
    def is_initialized(self) -> bool:
        """Check if model is loaded."""
        return self.intent_classifier.is_initialized()
