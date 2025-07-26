"""
The `ai` package provides the core intelligence for the command-line assistant,
encapsulating the agent, its environment, and the LLM client.
"""

from .agent import Agent, Environment
from .assistants.do import do
from .assistants.explain import explain

__all__ = [
    "Agent",
    "Environment",
    "do",
    "explain",
]