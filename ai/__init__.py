"""
The `ai` package provides the core intelligence for the command-line assistant,
encapsulating the agent, its environment, and the LLM client.
"""

from .agent import Agent, Environment, InteractiveEnvironment, MyEnvironment
from .assistant import get_shell_command

__all__ = [
    "Agent",
    "Environment",
    "InteractiveEnvironment",
    "MyEnvironment",
    "get_shell_command",
]