# ai/agents/wellness_agent.py
from typing import Literal
from enum import Enum
from ai.chains.base_chain import BaseChain

class WellnessState(Enum):
    STRESSED = "stressed"
    ANXIOUS = "anxious"
    CALM = "calm"
    UNKNOWN = "unknown"

class WellnessAgent(BaseChain):
    """
    Mental health support agent with:
    - Sentiment analysis
    - Stress detection
    - Personalized coping strategies
    """
    
    def __init__(self):
        super().__init__()
        self.coping_strategies = {
            WellnessState.STRESSED: [
                "Try the 5-4-3-2-1 grounding technique",
                "Take a 5-minute walk outside",
                "Write down 3 things you're grateful for"
            ],
            WellnessState.ANXIOUS: [
                "Box breathing: Inhale 4s, hold 4s, exhale 4s",
                "Listen to calming music for 3 minutes",
                "Name 5 things you can see around you"
            ]
        }

    def analyze_mood(self, text: str) -> WellnessState:
        """Detect emotional state using LLM"""
        prompt = f"""Classify this student message into exactly one of these categories:
        - "stressed" (overwhelmed with work)
        - "anxious" (nervous about performance)
        - "calm" (positive/neutral)
        
        Message: "{text}"
        
        Respond ONLY with one of the specified words."""
        
        response = self.llm.invoke(prompt).strip().lower()
        
        try:
            return WellnessState(response)
        except ValueError:
            return WellnessState.UNKNOWN

    def generate_response(self, state: WellnessState) -> str:
        """Provide personalized mental health support"""
        if state == WellnessState.CALM:
            return "You're doing great! Keep up the good work 🌟"
            
        strategies = self.coping_strategies.get(state, [])
        strategy = strategies[hash(text) % len(strategies)] if strategies else "Take 3 deep breaths"
        
        return (
            f"I notice you're feeling {state.value}. Here's something that might help:\n"
            f"✨ {strategy}\n"
            f"Remember: It's okay to take breaks."
        )

    def full_pipeline(self, text: str) -> str:
        """End-to-end wellness check"""
        state = self.analyze_mood(text)
        return self.generate_response(state)

# Test implementation
if __name__ == "__main__":
    agent = WellnessAgent()
    
    test_cases = [
        "I'm so overwhelmed with finals",
        "What if I fail this exam?",
        "I feel prepared for my presentation"
    ]
    
    for msg in test_cases:
        print(f"Input: {msg}")
        print(agent.full_pipeline(msg))
        print("-" * 50)