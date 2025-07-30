"""
The `ai` package provides the core intelligence for the command-line assistant,
encapsulating the agent, its environment, and the LLM client.
"""

from .agent import Agent, Environment
from .assistants.do import do
from .assistants.explain import explain
from .assistants.man import man
from .assistants.summarize import summarize
from .assistants.boilerplate import boilerplate
from .assistants.readmify import readmify
from .assistants.chat import chat


__all__ = [
    "Agent",
    "Environment",
    "do",
    "explain",
    "man",
    "summarize",
    "boilerplate",
    "readmify",
    "chat",
]