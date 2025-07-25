"""
The `ai` package provides the core intelligence for the command-line assistant,
encapsulating the agent, its environment, and the LLM client.
"""

from .agent import Agent, Environment, InteractiveEnvironment, MyEnvironment
from .assistants.do import do

__all__ = [
    "Agent",
    "Environment",
    "do",
]