# ai/agents/__init__.py
"""
Agent Module Exports
Exposes all agent classes for clean imports like:
from ai.agents import QAAgent, WellnessAgent, SchedulerAgent
"""

from .qa_agent import QAAgent
from .wellness_agent import WellnessAgent
from .scheduler_agent import SchedulerAgent

# Optional: Version tracking
__version__ = "1.0.0"

# Explicit exports control
__all__ = [
    "QAAgent",
    "WellnessAgent",
    "SchedulerAgent"
]

# Optional: Package-level initialization
def init_agents():
    """Pre-load models or verify dependencies"""
    from ai.tools.ollama_manager import OllamaManager
    OllamaManager().verify_models()